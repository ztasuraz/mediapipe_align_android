package com.google.mediapipe.examples.facelandmarker

import android.graphics.Bitmap

class ImageStack {
    private val frameTimeList = mutableListOf<Long>()
    private val frameTimeMap = mutableMapOf<Long, Bitmap>()

    // Add a pair of bmpImage and frameTime to the collection
    fun addFrame(frameTime: Long, bmpImage: Bitmap) {
        frameTimeList.add(frameTime)
        frameTimeMap[frameTime] = bmpImage
    }

    // Retrieve and remove a bmpImage by frameTime
    fun getFrameByTime(frameTime: Long): Bitmap? {
        val bmpImage = frameTimeMap.remove(frameTime)
        frameTimeList.remove(frameTime)
        return bmpImage
    }

    // Get the list of frameTimes in the order they were added
    fun getFrameTimes(): List<Long> {
        return frameTimeList
    }

    // Remove frames older than the specified time in milliseconds
    fun cleanupOldFrames(maxAgeMillis: Long) {
        val currentTime = System.currentTimeMillis()
        val iterator = frameTimeList.iterator()

        while (iterator.hasNext()) {
            val frameTime = iterator.next()
            if (currentTime - frameTime > maxAgeMillis) {
                // Remove outdated entry from the map and the list
                frameTimeMap.remove(frameTime)
                iterator.remove()
            }
        }
    }
}