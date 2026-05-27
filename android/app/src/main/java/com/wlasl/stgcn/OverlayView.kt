package com.wlasl.stgcn

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.min

class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var poseResult: PoseLandmarkerResult? = null
    private var handResult: HandLandmarkerResult? = null
    var label: String = "Waiting..."; private set
    var confidence: Float = 0f; private set

    // Input image dimensions from the camera analyzer — needed to map normalized
    // landmark coords (relative to input image) to screen coords correctly.
    private var imgW: Int = 1
    private var imgH: Int = 1

    private val bonePaint = Paint().apply {
        color = Color.GREEN; strokeWidth = 6f; style = Paint.Style.STROKE; isAntiAlias = true
    }
    private val jointPaint = Paint().apply {
        color = Color.GREEN; style = Paint.Style.FILL
    }
    private val handPaint = Paint().apply {
        color = Color.CYAN; strokeWidth = 5f; style = Paint.Style.STROKE; isAntiAlias = true
    }
    private val bgPaint = Paint().apply {
        color = Color.argb(190, 0, 0, 0); style = Paint.Style.FILL
    }
    private val labelPaint = Paint().apply {
        color = Color.GREEN; textSize = 60f; typeface = Typeface.DEFAULT_BOLD; isAntiAlias = true
    }
    private val confPaint = Paint().apply {
        color = Color.WHITE; textSize = 38f; isAntiAlias = true
    }

    fun setImageSize(w: Int, h: Int) {
        imgW = w; imgH = h
    }

    fun update(
        pose: PoseLandmarkerResult?,
        hand: HandLandmarkerResult?,
        lbl: String,
        conf: Float
    ) {
        poseResult = pose
        handResult = hand
        label = lbl
        confidence = conf
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val viewW = width.toFloat()
        val viewH = height.toFloat()

        // Fit-center scale: maintain aspect ratio of input image inside the view.
        // MediaPipe normalizes landmarks to [0,1] relative to the input image,
        // so we must map back to the same rectangle the preview occupies on screen.
        val scale = min(viewW / imgW, viewH / imgH)
        val offsetX = (viewW - imgW * scale) / 2f
        val offsetY = (viewH - imgH * scale) / 2f

        fun lx(nx: Float) = nx * imgW * scale + offsetX
        fun ly(ny: Float) = ny * imgH * scale + offsetY

        poseResult?.landmarks()?.firstOrNull()?.let { lms ->
            for ((a, b) in POSE_CONNECTIONS) {
                if (a < lms.size && b < lms.size) {
                    canvas.drawLine(lx(lms[a].x()), ly(lms[a].y()), lx(lms[b].x()), ly(lms[b].y()), bonePaint)
                }
            }
            for (lm in lms) {
                canvas.drawCircle(lx(lm.x()), ly(lm.y()), 8f, jointPaint)
            }
        }

        handResult?.landmarks()?.forEach { lms ->
            for ((a, b) in HAND_CONNECTIONS) {
                if (a < lms.size && b < lms.size) {
                    canvas.drawLine(lx(lms[a].x()), ly(lms[a].y()), lx(lms[b].x()), ly(lms[b].y()), handPaint)
                }
            }
        }

        canvas.drawRect(0f, 0f, viewW, 130f, bgPaint)
        canvas.drawText(label.uppercase(), 24f, 88f, labelPaint)
        canvas.drawText("%.1f%%".format(confidence * 100f), viewW - 180f, 88f, confPaint)
    }

    companion object {
        private val POSE_CONNECTIONS = listOf(
            11 to 12, 11 to 23, 12 to 24, 23 to 24,
            12 to 14, 14 to 16,
            11 to 13, 13 to 15,
            24 to 26, 26 to 28,
            23 to 25, 25 to 27,
            0 to 1, 1 to 2, 2 to 3, 3 to 7,
            0 to 4, 4 to 5, 5 to 6, 6 to 8,
            9 to 10,
            15 to 17, 15 to 19, 15 to 21,
            16 to 18, 16 to 20, 16 to 22
        )
        private val HAND_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            0 to 9, 9 to 10, 10 to 11, 11 to 12,
            0 to 13, 13 to 14, 14 to 15, 15 to 16,
            0 to 17, 17 to 18, 18 to 19, 19 to 20,
            5 to 9, 9 to 13, 13 to 17
        )
    }
}
