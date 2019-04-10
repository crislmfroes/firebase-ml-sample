package com.crislmfroes.firebasemlsample

import android.graphics.*
import android.os.Build
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.LinearLayout
import android.widget.TextView
import com.google.firebase.ml.common.modeldownload.FirebaseCloudModelSource
import com.google.firebase.ml.common.modeldownload.FirebaseLocalModelSource
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.custom.*
import io.fotoapparat.Fotoapparat
import io.fotoapparat.characteristic.LensPosition
import io.fotoapparat.log.logcat
import io.fotoapparat.parameter.ScaleType
import io.fotoapparat.preview.Frame
import io.fotoapparat.selector.*
import io.fotoapparat.view.CameraView
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.experimental.and

class MainActivity : AppCompatActivity() {

    private var camera : CameraView? = null
    private var fotoapparat : Fotoapparat? = null

    private var mapPredictions : HashMap<String, Float>? = null
    private val labels = mutableListOf<String>()

    private var manager : FirebaseModelManager? = null
    private var interpreter : FirebaseModelInterpreter? = null
    private var inOutOptions : FirebaseModelInputOutputOptions? = null

    private var canProcess = true

    private var linearLayout : LinearLayout? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mapPredictions = loadLabels("labels.txt")
        initFirebaseML()
        Log.d("TAG", "Firebase iniciado com sucesso!")
        setContentView(R.layout.activity_main)
        camera = findViewById(R.id.camera_view)
        linearLayout = findViewById(R.id.linearLayout)
        fotoapparat = Fotoapparat
            .with(this)
            .into(camera!!)
            .frameProcessor {
                processFrame(it)
            }
            .build()
    }

    override fun onStart() {
        super.onStart()
        fotoapparat!!.start()
    }

    override fun onStop() {
        super.onStop()
        fotoapparat!!.stop()
    }

    private fun processFrame(frame : Frame) {
        if (canProcess) {
            canProcess = false
            val image = YuvImage(frame.image, ImageFormat.NV21, frame.size.width, frame.size.height, null)
            val out = ByteArrayOutputStream()
            image.compressToJpeg(Rect(0, 0, image.width, image.height), 50, out)
            val imageBytes = out.toByteArray()
            val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val input = Array(1) {Array(224) {Array(224) {ByteArray(3)}}}
            for (x in 0..223) {
                for (y in 0..223) {
                    val pixel = scaledBitmap.getPixel(x, y)
                    Log.d("ei", (pixel shr 16 and 0xff).toString())
                    input[0][x][y][0] = (pixel shr 16 and 0xff).toByte()
                    input[0][x][y][1] = (pixel shr 8 and 0xff).toByte()
                    input[0][x][y][2] = (pixel and 0xff).toByte()
                }
            }
            val inputs = FirebaseModelInputs.Builder()
                .add(input)
                .build()
            interpreter!!.run(inputs, inOutOptions!!)
                .addOnSuccessListener {
                    val output = it.getOutput<Array<ByteArray>>(0)
                    val probabilities = output[0]
                    processPredictions(probabilities)
                    canProcess = true
                }
                .addOnFailureListener {
                    it.printStackTrace()
                    canProcess = true
                }
        }
    }

    private fun loadLabels(path : String) : HashMap<String, Float> {
        val lines = assets.open(path).reader().readLines()
        val mapLabels = hashMapOf<String, Float>()
        for (line in lines) {
            mapLabels[line.trim()] = 0.0f
            labels.add(line.trim())
        }
        return mapLabels
    }

    private fun initFirebaseML() {
        var conditionsBuilder = FirebaseModelDownloadConditions.Builder().requireWifi()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            conditionsBuilder = conditionsBuilder
                .requireCharging()
                .requireDeviceIdle()
        }
        val conditions = conditionsBuilder.build()
        val cloudSource = FirebaseCloudModelSource.Builder("mobilenet_v1")
            .enableModelUpdates(true)
            .setInitialDownloadConditions(conditions)
            .setUpdatesDownloadConditions(conditions)
            .build()
        manager = FirebaseModelManager.getInstance()
        manager!!.registerCloudModelSource(cloudSource)
        val localSource = FirebaseLocalModelSource.Builder("mobilenet_v1")
            .setAssetFilePath("mobilenet_v1.tflite")
            .build()
        manager!!.registerLocalModelSource(localSource)
        val options = FirebaseModelOptions.Builder()
            .setCloudModelName("mobilenet_v1")
            .setLocalModelName("mobilenet_v1")
            .build()
        interpreter = FirebaseModelInterpreter.getInstance(options)

        inOutOptions = FirebaseModelInputOutputOptions.Builder()
            .setInputFormat(0, FirebaseModelDataType.BYTE, intArrayOf(1, 224, 224, 3))
            .setOutputFormat(0, FirebaseModelDataType.BYTE, intArrayOf(1, 1001))
            .build()
    }

    private fun processPredictions(predictions : ByteArray) {
        for (i in 0 until predictions.size - 1) {
            mapPredictions!![labels[i]] = predBytesToFloat(predictions[i])
        }
        val sortedLabels = mapPredictions!!.toList().sortedByDescending {
            it.second
        }
        linearLayout!!.removeAllViews()
        for (i in 0..3) {
            val pair = sortedLabels[i]
            val text = "%s:%f".format(pair.first, pair.second)
            val textView = TextView(this.applicationContext)
            textView.text = text
            linearLayout!!.addView(textView)
        }
    }

    private fun predBytesToFloat(prediction : Byte) : Float {
        return (prediction.toInt() and 0xff) / 255.0f
    }

}
