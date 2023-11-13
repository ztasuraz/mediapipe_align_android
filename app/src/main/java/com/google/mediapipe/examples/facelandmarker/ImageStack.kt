package com.google.mediapipe.examples.facelandmarker

import android.graphics.Bitmap
import android.os.SystemClock

class ImageStack {
    private val frameTimeList = mutableListOf<Long>()
    private val frameTimeMap = mutableMapOf<Long, Bitmap>()

    // Add a pair of bmpImage and frameTime to the collection
    fun addFrame(frameTime: Long, bmpImage: Bitmap) {
        synchronized(this) {
            frameTimeList.add(frameTime)
            frameTimeMap[frameTime] = bmpImage
        }
    }

    // Retrieve and remove a bmpImage by frameTime
    fun getFrameByTime(frameTime: Long): Bitmap? {
        synchronized(this) {
            val bmpImage = frameTimeMap.remove(frameTime)
            frameTimeList.remove(frameTime)
            return bmpImage
        }
    }

    // Get the list of frameTimes in the order they were added
    fun getFrameTimes(): List<Long> {
        synchronized(this) {
            return frameTimeList.toList()
        }
    }

    // Remove frames older than the specified time in milliseconds
    fun cleanupOldFrames(maxAgeMillis: Long) {
        val currentTime = SystemClock.uptimeMillis()

        synchronized(this) {
            val iterator = frameTimeList.iterator()

            while (iterator.hasNext()) {
                val frameTime = iterator.next()
                val debug = currentTime - frameTime
                if (currentTime - frameTime > maxAgeMillis) {
                    // Remove outdated entry from the map and the list
                    frameTimeMap.remove(frameTime)
                    iterator.remove()
                }
            }
        }
    }
}
