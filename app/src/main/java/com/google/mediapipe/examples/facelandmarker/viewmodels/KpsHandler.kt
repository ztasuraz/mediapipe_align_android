package com.google.mediapipe.examples.facelandmarker.viewmodels
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
//import android.graphics.BitmapFactory
import android.util.Log
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.calib3d.Calib3d.estimateAffinePartial2D
import org.opencv.calib3d.Calib3d.LMEDS
import org.opencv.imgproc.Imgproc.warpAffine
import org.opencv.android.Utils.bitmapToMat
import org.opencv.android.Utils.matToBitmap
import org.opencv.core.MatOfInt
import org.opencv.core.Size
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.opencv.calib3d.Calib3d.decomposeProjectionMatrix
import org.opencv.core.CvType
import org.opencv.core.CvType.CV_32F

class KpsHandler {

    private val points = arrayOf(
        Point(38.2946, 51.6963),
        Point(73.5318, 51.5014),
        Point(56.0252, 71.7366),
        Point(41.5493, 92.3655),
        Point(70.7299, 92.2041)
    )

    // Create a new MatOfPoint2f and initialize it with the points
    private var refKpsFull: MatOfPoint2f ?= null
    private var refKpsHalf: MatOfPoint2f ?= null
    private val rightMouth = 291
    private val leftMouth = 61
    private val nose = 1
    private var rightEye: MatOfInt ?= null
    private var leftEye: MatOfInt ?= null

    init {
        if (!OpenCVLoader.initDebug())
            Log.e("OpenCV", "Unable to load OpenCV!")
        else {

            rightEye = MatOfInt(
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398
            )
            leftEye = MatOfInt(
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246
            )
            refKpsFull = MatOfPoint2f()
            refKpsHalf = MatOfPoint2f()
            refKpsFull!!.fromArray(*points)
//            for ((index, point) in points.withIndex()) {
//                Log.d("PointLog", "refKpsFull $index: x = ${point.x}, y = ${point.y}")
//            }
            val meanPoint = Point(
                (points[points.size - 1].x + points[points.size - 2].x) / 2.0,
                (points[points.size - 1].y + points[points.size - 2].y) / 2.0
            )
            val halfPoints = arrayOf(
                points[0],
                points[1],
                points[2],
                meanPoint
            )
            refKpsHalf!!.fromArray(*halfPoints)
//            for ((index, point) in halfPoints.withIndex()) {
//                Log.d("PointLog", "refKpsHalf $index: x = ${point.x}, y = ${point.y}")
//            }
            Log.d("OpenCV", "OpenCV loaded Successfully!")
        }
    }

    fun rescaleKps(landmarks: MatOfPoint2f, originalScale: Pair<Double, Double>, inputScale: Pair<Double, Double>): MatOfPoint2f {

        val originalWidth = originalScale.first
        val originalHeight = originalScale.second
        val newWidth = inputScale.first
        val newHeight = inputScale.second

        val widthScaleFactor = originalWidth / newWidth
        val heightScaleFactor = originalHeight / newHeight
        val tempList : MutableList<Point> = ArrayList()
        val normalizedLandmarks = MatOfPoint2f()

        for (point in landmarks.toList()) {
            val normalizedX = point.x * widthScaleFactor
            val normalizedY = point.y * heightScaleFactor
            tempList.add(Point(normalizedX, normalizedY))
        }
        normalizedLandmarks.fromList(tempList)

        return normalizedLandmarks
    }

    fun normCrop(img: Mat, lmk: MatOfPoint2f, full: Boolean, size: Double = 112.0): Mat {
        val inlier = Mat()

        val m: Mat = if (!full) {
            estimateAffinePartial2D(lmk, refKpsHalf, inlier, LMEDS)
        } else {
            estimateAffinePartial2D(lmk, refKpsFull, inlier, LMEDS)
        }
        val crop = Mat()
        // Ensure img and m are of data type CV_32F
//        if (img.type() != CvType.CV_32F) {
//            img.convertTo(img, CvType.CV_32F)
//        }
//        if (m.type() != CvType.CV_32F) {
//            m.convertTo(m, CvType.CV_32F)
//        }

        // Log the shape and data type of m
        Log.d(
            "NormCrop",
            "Shape of m: ${m.rows()} x ${m.cols()}, Data Type: ${CvType.typeToString(m.type())}"
        )

        // Log the shape and data type of img
        Log.d(
            "NormCrop",
            "Shape of img: ${img.rows()} x ${img.cols()}, Data Type: ${CvType.typeToString(img.type())}"
        )

        // Perform affine
        warpAffine(img, crop, m, Size(size, size))

        // Log the shape and data type of crop
        Log.d(
            "NormCrop",
            "Shape of crop: ${crop.rows()} x ${crop.cols()}, Data Type: ${CvType.typeToString(crop.type())}"
        )

        return crop
    }

    private fun getCenterEye(eyeKps: MutableList<Point>): Point {
        if (eyeKps.isEmpty()) {
            // Handle the case where the list of keypoints is empty or contains no keypoints.
            // You can return a default or special point, or handle it as needed.
            return Point(-1.0, -1.0) // Return a point with negative coordinates as an example.
        }

        var sumX = 0.0
        var sumY = 0.0

        // Calculate the sum of x and y coordinates
        for (point in eyeKps) {
            sumX += point.x
            sumY += point.y
        }

        // Calculate the average (center) point
        val centerX = sumX / eyeKps.size
        val centerY = sumY / eyeKps.size

        return Point(centerX, centerY)
    }

    private fun printModifiedKps(modifiedKps: MatOfPoint2f) {
        val points = modifiedKps.toList()
        for ((index, point) in points.withIndex()) {
            Log.d("PointLog", "modifiedKps $index: x = ${point.x}, y = ${point.y}")
        }
    }

    private fun scaleAndClipNormalizedPoints(
        points2D: MutableList<Point>,
        width: Int,
        height: Int
    ): Pair<MutableList<Point>, Boolean> {
        val scaledPoints = mutableListOf<Point>()
        var exceedsDimensions = false

        for (point in points2D) {
            val scaledX = point.x * width
            val scaledY = point.y * height

            // Clip the scaled coordinates to ensure they stay within the frame
            val clippedX = scaledX.coerceIn(0.0, width.toDouble())
            val clippedY = scaledY.coerceIn(0.0, height.toDouble())

            val clippedPoint = Point(clippedX, clippedY)
            scaledPoints.add(clippedPoint)

            // Check if the point exceeds the dimensions
            if (scaledX < 0.0 || scaledX > width.toDouble() || scaledY < 0.0 || scaledY > height.toDouble()) {
                exceedsDimensions = true
            }

            // Log the scaled and clipped point
            Log.d("PointLog", "Original Point: x = ${point.x}, y = ${point.y}")
            Log.d("PointLog", "Scaled & Clipped Point: x = ${clippedPoint.x}, y = ${clippedPoint.y}")
        }

        return Pair(scaledPoints, exceedsDimensions)
    }



    fun drawPointsOnBitmap(bmpImage: Bitmap, points2D: MatOfPoint2f): Bitmap {
        val imageWithPoints = bmpImage.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(imageWithPoints)
        val paint = Paint()
        paint.color = Color.RED // Set point color (you can change this)

        // Get the width and height of the Bitmap
        val width = bmpImage.width.toFloat()
        val height = bmpImage.height.toFloat()

        // Convert the MatOfPoint2f to a list of Point
        val pointsList = points2D.toList()

        // Define the radius for drawing points
        val pointRadius = 20.0f

        // Draw each point on the Bitmap
        for (point in pointsList) {
            val x = (point.x * width).toFloat()
            val y = (point.y * height).toFloat()
            canvas.drawCircle(x, y, pointRadius, paint)
        }

        return imageWithPoints
    }

    fun parseKps(kps: List<NormalizedLandmark>, width: Int, height: Int, full: Boolean): Pair<MatOfPoint2f, Boolean> {
        val modifiedKps = MatOfPoint2f()
        //  Loop through the list of NormalizedLandmark
        //  and add them to the MatOfPoint2f
        val points2D : MutableList<Point> = ArrayList()
        if (!full) {
            for (i in kps.indices) {
                // break if i > 3, since we only need 4 points
                if (i > 3) break
                points2D.add(Point(kps[i].x().toDouble(), kps[i].y().toDouble()))
            }
        } else {
            val rEyeKps : MutableList<Point> = ArrayList()
            for (i in rightEye?.toList()!!) {
                rEyeKps.add(Point(kps[i].x().toDouble(), kps[i].y().toDouble()))
            }
            val lEyeKps : MutableList<Point> = ArrayList()
            for (i in leftEye?.toList()!!) {
                lEyeKps.add(Point(kps[i].x().toDouble(), kps[i].y().toDouble()))
            }
            points2D.add(getCenterEye(lEyeKps))
            points2D.add(getCenterEye(rEyeKps))
            points2D.add(Point(kps[nose].x().toDouble(), kps[nose].y().toDouble()))
            points2D.add(Point(kps[leftMouth].x().toDouble(), kps[leftMouth].y().toDouble()))
            points2D.add(Point(kps[rightMouth].x().toDouble(), kps[rightMouth].y().toDouble()))
        }
        val (scaledPoints2D, exceedsDimensions) = scaleAndClipNormalizedPoints(points2D, width, height)

        modifiedKps.fromList(scaledPoints2D)
        printModifiedKps(modifiedKps)
        return Pair(modifiedKps, exceedsDimensions)
    }

    fun parseTransMatrix(floatArrays: FloatArray): FloatArray {
        // Convert to cv array
        val poseMat = Mat(4, 4, CV_32F)
        poseMat.put(0, 0, floatArrays)


        // Log the poseMat for debugging
        println("Debug - poseMat:\n${poseMat.dump()}")


        // Transpose for proper decomposition
        val submatrix = Mat(poseMat, org.opencv.core.Rect(0, 0, 3, 4)).t()

        println("Debug - submatrix:\n${submatrix.dump()}")
//        // Decompose projection matrix
        val cameraMatrix = Mat()
        val rotMatrix = Mat()
        val transVect = Mat()
        val rotMatrixX = Mat()
        val rotMatrixY = Mat()
        val rotMatrixZ = Mat()
        val eulerAngles = Mat()
        decomposeProjectionMatrix(submatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles)
        println("Debug - submatrix:\n${eulerAngles.dump()}")
        // Convert Mat to FloatArray
        val eulerAnglesArray = FloatArray(eulerAngles.cols() * eulerAngles.rows())
        eulerAngles.convertTo(eulerAngles, CV_32F)
        eulerAngles.get(0, 0, eulerAnglesArray)

        // Print the converted Euler angles for debugging
        println("Debug - Euler Angles Array: ${eulerAnglesArray.contentToString()}")

        return eulerAnglesArray
    }

    // Function to convert from bitmap to mat
    fun parseImg(img: Bitmap): Mat {
        val mat = Mat()
        val bmp32 = img.copy(Bitmap.Config.ARGB_8888, true)
        bitmapToMat(bmp32, mat)
        return mat
    }



    fun reparseImg(img: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(img.cols(), img.rows(), Bitmap.Config.ARGB_8888)
        matToBitmap(img, bmp)
        return bmp
    }
}


//fun main() {
//    /*
//    Example usage
//    */
//    val kps = listOf(
//        NormalizedLandmark(78.9359f, 80.2342f, 0.0f),
//        NormalizedLandmark(106.1096f, 109.7709f, 0.0f),
//        NormalizedLandmark(75.7302f, 108.6659f, 0.0f),
//        NormalizedLandmark(60.9624f, 123.9752f, 0.0f),
//        NormalizedLandmark(57.2560f, 74.2407f, 0.0f),
//        NormalizedLandmark(114.2049f, 134.4578f, 0.0f)
//    )
//    // print length of kps
//    println(kps.size)
//    val handler = KpsHandler()
//    // convert from List<NormalizedLandmark> to MatOfPoint2f
//    val transformedKps = handler.parseKps(kps, full = false)
//    // Test img
//    val img = BitmapFactory.decodeFile("test.jpg")
//    // BMP to MAT
//    val transformedImg = handler.parseImg(img)
//    // Alignment
//    val alignedImg = handler.normCrop(transformedImg, transformedKps, full = false)
//    // Mat to BMP
//    val alignedImgBmp = handler.reparseImg(alignedImg)
////    for
////    println(transformedKps)
//}