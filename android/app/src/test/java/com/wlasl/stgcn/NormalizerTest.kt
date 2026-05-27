package com.wlasl.stgcn

import org.junit.Assert.*
import org.junit.Test

class NormalizerTest {

    private val norm = LandmarkNormalizer(60)

    @Test
    fun `nose centered to origin after spatial normalization`() {
        val frame = FloatArray(75 * 3)
        frame[0] = 0.5f; frame[1] = 0.6f; frame[2] = 0.1f  // nose (joint 0)
        frame[33] = 0.4f; frame[34] = 0.5f; frame[35] = 0.0f  // left shoulder (joint 11)
        frame[36] = 0.6f; frame[37] = 0.5f; frame[38] = 0.0f  // right shoulder (joint 12)

        val result = norm.process(Array(60) { if (it == 0) frame else FloatArray(75 * 3) })

        assertEquals(0f, result[0][0], 1e-5f)
        assertEquals(0f, result[0][1], 1e-5f)
        assertEquals(0f, result[0][2], 1e-5f)
    }

    @Test
    fun `short buffer zero-pads to 60 frames`() {
        val frames = Array(30) { FloatArray(75 * 3) { 1f } }
        val result = norm.process(frames)

        assertEquals(60, result.size)
        for (i in 30 until 60) assertTrue(result[i].all { it == 0f })
    }

    @Test
    fun `long buffer subsampled to 60 frames`() {
        val frames = Array(120) { FloatArray(75 * 3) }
        val result = norm.process(frames)
        assertEquals(60, result.size)
    }

    @Test
    fun `exact 60 frames passes through unchanged`() {
        val frames = Array(60) { t -> FloatArray(75 * 3) { t.toFloat() } }
        val result = norm.process(frames)
        assertEquals(60, result.size)
        assertArrayEquals(frames[0], result[0], 1e-7f)
        assertArrayEquals(frames[59], result[59], 1e-7f)
    }

    @Test
    fun `frame with zero nose skipped in spatial normalization`() {
        val frame = FloatArray(75 * 3)
        // nose = (0,0,0) → skip this frame, no centering
        frame[3] = 0.5f  // joint 1 x
        val result = norm.process(Array(60) { if (it == 0) frame.copyOf() else FloatArray(75 * 3) })
        // joint 1 x should be unchanged (0.5) since nose was zero
        assertEquals(0.5f, result[0][3], 1e-5f)
    }
}
