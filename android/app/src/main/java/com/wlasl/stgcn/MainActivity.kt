package com.wlasl.stgcn

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var poseLandmarker: PoseLandmarker
    private lateinit var handLandmarker: HandLandmarker
    private lateinit var inferenceManager: SignInferenceManager

    // Pose callback fires first; hand result is best-effort from last frame
    @Volatile private var latestHandResult: HandLandmarkerResult? = null
    @Volatile private var currentLabel: String = "Waiting..."
    @Volatile private var currentConfidence: Float = 0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview_view)
        overlayView = findViewById(R.id.overlay_view)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            init()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
        }
    }

    override fun onRequestPermissionsResult(req: Int, perms: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(req, perms, results)
        if (req == REQ_CAMERA && results.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            init()
        }
    }

    private fun init() {
        inferenceManager = SignInferenceManager(this)
        inferenceManager.onResult = { label, conf ->
            currentLabel = label
            currentConfidence = conf
        }
        setupMediaPipe()
        startCamera()
    }

    private fun setupMediaPipe() {
        poseLandmarker = PoseLandmarker.createFromOptions(
            this,
            PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").build()
                )
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setResultListener { result: PoseLandmarkerResult, _ -> onPoseResult(result) }
                .setErrorListener { e -> Log.e(TAG, "Pose error: $e") }
                .build()
        )

        handLandmarker = HandLandmarker.createFromOptions(
            this,
            HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
                )
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setResultListener { result: HandLandmarkerResult, _ -> latestHandResult = result }
                .setErrorListener { e -> Log.e(TAG, "Hand error: $e") }
                .build()
        )
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setTargetRotation(previewView.display?.rotation ?: android.view.Surface.ROTATION_0)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val timestampMs = imageProxy.imageInfo.timestamp / 1_000_000L

                // Rotate + mirror so MediaPipe receives an upright, front-camera-mirrored image.
                // Without this, landmark coordinates are in the raw sensor space (landscape, unmirrored).
                val raw = imageProxy.toBitmap()
                val matrix = android.graphics.Matrix().apply {
                    postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                    postScale(-1f, 1f, raw.width / 2f, raw.height / 2f)
                }
                val rotated = android.graphics.Bitmap.createBitmap(
                    raw, 0, 0, raw.width, raw.height, matrix, true
                )
                imageProxy.close()

                val mpImage = BitmapImageBuilder(rotated).build()
                poseLandmarker.detectAsync(mpImage, timestampMs)
                handLandmarker.detectAsync(mpImage, timestampMs)

                // Tell overlay the input dimensions so it can scale landmarks correctly
                runOnUiThread { overlayView.setImageSize(rotated.width, rotated.height) }
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onPoseResult(poseResult: PoseLandmarkerResult) {
        val handResult = latestHandResult
        val landmarks = buildLandmarkArray(poseResult, handResult)
        inferenceManager.onFrame(landmarks)

        runOnUiThread {
            overlayView.update(poseResult, handResult, currentLabel, currentConfidence)
        }
    }

    private fun buildLandmarkArray(
        pose: PoseLandmarkerResult,
        hand: HandLandmarkerResult?
    ): FloatArray {
        val flat = FloatArray(75 * 3)  // zeros = not detected

        // Pose joints 0-32
        pose.landmarks().firstOrNull()?.forEachIndexed { i, lm ->
            flat[i * 3]     = lm.x()
            flat[i * 3 + 1] = lm.y()
            flat[i * 3 + 2] = lm.z()
        }

        // Left hand: joints 33-53, Right hand: joints 54-74
        hand?.let { h ->
            h.landmarks().zip(h.handedness()).forEach { (lms, handed) ->
                val offset = if (handed.first().categoryName() == "Left") 33 else 54
                lms.forEachIndexed { i, lm ->
                    flat[(offset + i) * 3]     = lm.x()
                    flat[(offset + i) * 3 + 1] = lm.y()
                    flat[(offset + i) * 3 + 2] = lm.z()
                }
            }
        }

        return flat
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (::poseLandmarker.isInitialized) poseLandmarker.close()
        if (::handLandmarker.isInitialized) handLandmarker.close()
        if (::inferenceManager.isInitialized) inferenceManager.close()
    }

    companion object {
        private const val TAG = "WLASL"
        private const val REQ_CAMERA = 1001
    }
}
