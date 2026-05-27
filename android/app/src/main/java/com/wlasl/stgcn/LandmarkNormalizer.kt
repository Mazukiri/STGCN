package com.wlasl.stgcn

import kotlin.math.sqrt

/**
 * Exact port of src/data_processing/normalization.py: LandmarkNormalizer.
 *
 * Input/output: Array of FloatArray(225) — 75 joints × 3 coords, flat as [x0,y0,z0, x1,y1,z1, ...].
 * Array length = T frames.
 */
class LandmarkNormalizer(private val targetFrames: Int = 60) {

    fun process(frames: Array<FloatArray>): Array<FloatArray> {
        val spatial = normalizeSpatial(frames)
        return normalizeTemporal(spatial)
    }

    private fun normalizeSpatial(frames: Array<FloatArray>): Array<FloatArray> {
        val result = Array(frames.size) { frames[it].copyOf() }
        for (t in result.indices) {
            val frame = result[t]
            val noseX = frame[0]; val noseY = frame[1]; val noseZ = frame[2]
            // Skip frames where pose was not detected
            if (noseX == 0f && noseY == 0f && noseZ == 0f) continue

            // Center all detected joints on nose (joint 0)
            for (j in 0 until 75) {
                val b = j * 3
                if (frame[b] != 0f || frame[b + 1] != 0f || frame[b + 2] != 0f) {
                    frame[b]     -= noseX
                    frame[b + 1] -= noseY
                    frame[b + 2] -= noseZ
                }
            }

            // Scale by shoulder distance (joints 11 and 12, already re-centered)
            val ls = 11 * 3; val rs = 12 * 3
            val dx = frame[ls] - frame[rs]
            val dy = frame[ls + 1] - frame[rs + 1]
            val dz = frame[ls + 2] - frame[rs + 2]
            val dist = sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()
            if (dist > 1e-5f) {
                for (j in 0 until 75) {
                    val b = j * 3
                    if (frame[b] != 0f || frame[b + 1] != 0f || frame[b + 2] != 0f) {
                        frame[b]     /= dist
                        frame[b + 1] /= dist
                        frame[b + 2] /= dist
                    }
                }
            }
        }
        return result
    }

    private fun normalizeTemporal(frames: Array<FloatArray>): Array<FloatArray> {
        val T = frames.size
        val zeros = FloatArray(75 * 3)
        return when {
            T == targetFrames -> frames
            T < targetFrames  -> Array(targetFrames) { i ->
                if (i < T) frames[i] else zeros.copyOf()
            }
            else -> Array(targetFrames) { i ->
                // Uniform sampling matching np.linspace(0, T-1, targetFrames, dtype=int)
                val srcIdx = (i.toFloat() * (T - 1) / (targetFrames - 1)).toInt()
                frames[srcIdx]
            }
        }
    }
}
