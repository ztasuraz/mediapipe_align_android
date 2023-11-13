package com.google.mediapipe.examples.facelandmarker

/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.max
import kotlin.math.min
import com.google.mediapipe.examples.facelandmarker.viewmodels.KpsHandler
class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: FaceLandmarkerResult? = null
    private var bmpImage: Bitmap ?= null
    private var linePaint = Paint()
    private var pointPaint = Paint()

    private var scaleFactor: Pair<Float, Float> = Pair(1f,1f)
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    private var handler = KpsHandler()
    init {
        initPaints()
    }

    fun clear() {
        results = null
        linePaint.reset()
        pointPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        if(results == null || results!!.faceLandmarks().isEmpty()) {
            clear()
            return
        }

        results?.let { faceLandmarkerResult ->

            if( faceLandmarkerResult.faceBlendshapes().isPresent) {
                faceLandmarkerResult.faceBlendshapes().get().forEach {
                    it.forEach {
                        Log.e(TAG, it.displayName() + " " + it.score())
                    }
                }
            }
            for(landmark in faceLandmarkerResult.faceLandmarks()) {
                for(normalizedLandmark in landmark) {
                    canvas.drawPoint(normalizedLandmark.x() * imageWidth * scaleFactor.first, normalizedLandmark.y() * imageHeight * scaleFactor.second, pointPaint)
//                    Log.d("PointLog", "Draw cord: x = ${normalizedLandmark.x()* imageWidth * scaleFactor.first}, y = ${normalizedLandmark.y() * imageHeight * scaleFactor.second}")
                }
            }
            FaceLandmarker.FACE_LANDMARKS_CONNECTORS.forEach {
                canvas.drawLine(
                    faceLandmarkerResult.faceLandmarks().get(0).get(it!!.start()).x() * imageWidth * scaleFactor.first,
                    faceLandmarkerResult.faceLandmarks().get(0).get(it.start()).y() * imageHeight * scaleFactor.second,
                    faceLandmarkerResult.faceLandmarks().get(0).get(it.end()).x() * imageWidth * scaleFactor.first,
                    faceLandmarkerResult.faceLandmarks().get(0).get(it.end()).y() * imageHeight * scaleFactor.second,
                    linePaint)
            }

            val transformedKps = handler.parseKps(faceLandmarkerResult.faceLandmarks()[0], imageWidth, imageHeight, true)
            val points = transformedKps.toList()
            for ((index, point) in points.withIndex()) {
                canvas.drawPoint(point.x.toFloat() * scaleFactor.first, point.y.toFloat() * scaleFactor.second, pointPaint)
//                Log.d("PointLog", "modifiedKps $index: x = ${point.x}, y = ${point.y}")
            }
            val transformedTransMatrix = handler.parseTransMatrix(results!!.facialTransformationMatrixes().get()[0])
            val (pitch, yaw, roll) = transformedTransMatrix.sliceArray(0 until 3)

            var straightFace = true
            Log.d("FacePoseRevised", "Yaw: ${yaw}, Pitch: ${pitch}, Roll: $roll")
            when {
                yaw.toInt() in -15..20 && roll.toInt() in -20..20  && pitch.toInt() in -15..15 -> {
                    Log.d("HeadPoseEstimation", "Face is straight")
                }
                else -> {
                    straightFace = false
                    Log.d("HeadPoseEstimation", "Face is not straight")
                }
            }
            if (straightFace) {
                bmpImage?.let {
                    // not null do something
                    val transformedImg = KpsHandler().parseImg(bmpImage!!)
                    val alignedImg = KpsHandler().normCrop(transformedImg, transformedKps, full = true)
                    val bmpAlignedImg = KpsHandler().reparseImg(alignedImg)
                }
                // Send bmp image to mFace server...
            }
        }
    }

    fun setResults(
        faceLandmarkerResults: FaceLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE,
        image: Bitmap ?= null,
    ) {
        results = faceLandmarkerResults
        bmpImage = image
        this.imageHeight = imageHeight
        this.imageWidth = imageWidth
        val a = Pair(width, height)
        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                val scale = min(width * 1f / imageWidth, height * 1f / imageHeight)
                Pair(scale, scale)
//                Pair(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                val scale = max(width * 1f / imageWidth, height * 1f / imageHeight)
                Pair(scale, scale)
//                Pair(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 8F
        private const val TAG = "Face Landmarker Overlay"
    }
}
