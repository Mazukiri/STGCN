plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.wlasl.stgcn"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.wlasl.stgcn"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    // Prevent compression of large model files — MediaPipe requires uncompressed .task files
    androidResources {
        noCompress += listOf("task", "onnx", "tflite")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")

    // MediaPipe Tasks Vision (pose + hand landmarker)
    implementation("com.google.mediapipe:tasks-vision:0.10.14")

    // ONNX Runtime for Android (includes NNAPI execution provider)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")

    // CameraX
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")

    // Coroutines (background inference thread)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    testImplementation("junit:junit:4.13.2")
}
