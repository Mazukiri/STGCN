package com.wlasl.stgcn

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.nio.FloatBuffer

class SignInferenceManager(context: Context) {

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val labels: Map<Int, String>
    private val numClasses: Int
    private val normalizer = LandmarkNormalizer(TARGET_FRAMES)

    private val buffer = ArrayDeque<FloatArray>()
    private var noDetectionCount = 0

    var onResult: ((label: String, confidence: Float) -> Unit)? = null

    init {
        val opts = OrtSession.SessionOptions()
        try {
            opts.addNnapi()  // DSP acceleration on Snapdragon 778G
        } catch (e: OrtException) {
            Log.w(TAG, "NNAPI unavailable, using CPU: ${e.message}")
        }
        val modelBytes = context.assets.open("stgcn_quant.onnx").readBytes()
        session = ortEnv.createSession(modelBytes, opts)

        val json = context.assets.open("label_mapping.json").bufferedReader().readText()
        val obj = JSONObject(json)
        val mutable = mutableMapOf<Int, String>()
        obj.keys().forEach { key -> mutable[obj.getInt(key)] = key }
        labels = mutable
        numClasses = labels.size
    }

    /**
     * Call once per camera frame.
     * landmarks: FloatArray(225) — flat [x0,y0,z0, x1,y1,z1, ...] for 75 joints.
     * Pass null when MediaPipe detects no person.
     */
    fun onFrame(landmarks: FloatArray?) {
        val posePresent = landmarks != null &&
            (landmarks[0] != 0f || landmarks[1] != 0f || landmarks[2] != 0f)

        if (!posePresent) {
            if (++noDetectionCount >= RESET_THRESHOLD) {
                buffer.clear()
                noDetectionCount = 0
                onResult?.invoke("Waiting...", 0f)
            }
            return
        }

        noDetectionCount = 0
        if (buffer.size % 30 == 0) {
            // Log raw values every 30 frames to verify pose is detected correctly
            Log.d(TAG, "raw: nose=(${landmarks!![0]},${landmarks[1]}), " +
                "ls=(${landmarks[33]},${landmarks[34]}), rs=(${landmarks[36]},${landmarks[37]}), " +
                "buf=${buffer.size}")
        }
        buffer.addLast(landmarks!!)
        if (buffer.size > TARGET_FRAMES) buffer.removeFirst()

        if (buffer.size == TARGET_FRAMES) {
            // Require hand data in at least 20 of 60 frames — model trained with hands visible;
            // all-zero hand joints produce near-uniform uncertain predictions.
            val framesWithHands = buffer.count { frame ->
                (33 until 75).any { j -> frame[j * 3] != 0f }
            }
            if (framesWithHands < 20) {
                onResult?.invoke("Show hands...", 0f)
                return
            }
            val norm = normalizer.process(buffer.toTypedArray())
            runInference(norm)
        }
    }

    private fun runInference(frames: Array<FloatArray>) {
        // Debug: log normalized shoulder positions (should be ~±0.5, 1.0 if normalization is correct)
        val f0 = frames[30]  // middle frame
        Log.d(TAG, "norm: ls=(${f0[33]},${f0[34]}), rs=(${f0[36]},${f0[37]}), lh0=(${f0[99]},${f0[100]})")

        // Transpose (60, 75, 3) → flat (1, 3, 60, 75)
        // Target index: c * 60*75 + t * 75 + j
        val flat = FloatArray(3 * TARGET_FRAMES * 75)
        for (t in 0 until TARGET_FRAMES) {
            for (j in 0 until 75) {
                for (c in 0..2) {
                    flat[c * TARGET_FRAMES * 75 + t * 75 + j] = frames[t][j * 3 + c]
                }
            }
        }

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(flat),
            longArrayOf(1, 3, TARGET_FRAMES.toLong(), 75)
        )

        try {
            session.run(mapOf("input" to inputTensor)).use { result ->
                val outputTensor = result.get(0) as OnnxTensor
                val logits = FloatArray(numClasses)
                outputTensor.floatBuffer.rewind()
                outputTensor.floatBuffer.get(logits)

                val maxLogit = logits.maxOrNull()!!
                val expArr = FloatArray(numClasses) { Math.exp((logits[it] - maxLogit).toDouble()).toFloat() }
                val expSum = expArr.sum()
                val probs = FloatArray(numClasses) { expArr[it] / expSum }

                val bestIdx = probs.indices.maxByOrNull { probs[it] }!!
                val confidence = probs[bestIdx]
                val label = if (confidence > CONFIDENCE_THRESHOLD) labels[bestIdx] ?: "?" else "..."

                // Debug: log top-3 predictions
                val top3 = probs.indices.sortedByDescending { probs[it] }.take(3)
                Log.d(TAG, "top3: " + top3.joinToString { "${labels[it]}=${probs[it].times(100).toInt()}%" })

                onResult?.invoke(label, confidence)
            }
        } finally {
            inputTensor.close()
        }
    }

    fun close() {
        session.close()
        ortEnv.close()
    }

    companion object {
        private const val TAG = "SignInference"
        private const val TARGET_FRAMES = 60
        private const val RESET_THRESHOLD = 15
        private const val CONFIDENCE_THRESHOLD = 0.5f
    }
}
