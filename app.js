const express = require("express");
const faceapi = require("face-api.js");
require("dotenv").config();
const mongoose = require("mongoose");
const { Canvas, Image } = require("canvas");
const canvas = require("canvas");
const fileUpload = require("express-fileupload");
faceapi.env.monkeyPatch({ Canvas, Image });

const app = express();

app.use(
  fileUpload({
    useTempFiles: true,
  })
);

/*
  Branch = single-face-recognizer
/*
Load Models
1. Memuat model dari faceRecognitionNet yang bertugas untuk pengenalan dan identifikasi wajah pada gambar yang diberikan
2. Memuat model dari faceLandmark68Net yang mampu mendeteksi 68 landmark atau titik kunci pada wajah seperti hidung, mata, mulut, dan lainnya.
3. Memuat model dari ssdMobilenetv1 yang berfungsi deteksi objek dan juga menemukan berbagai objek, termasuk wajah dalam sebuah gambar
 - faceapi.nets merupakan namespace yang disediakan oleh library 'face-api.js' yang berisikan model-model yang sudah terlatih dari folder "models"
*/
async function LoadModels() {
  await faceapi.nets.faceRecognitionNet.loadFromDisk(__dirname + "/models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk(__dirname + "/models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(__dirname + "/models");
}
LoadModels();

/*
Membuat collection baru pada MongoDB
*/
const faceSchema = new mongoose.Schema({
  label: {
    type: String,
    required: true,
    unique: true,
  },
  descriptions: {
    type: Array,
    required: true,
  },
});

const FaceModel = mongoose.model("Face", faceSchema);

/*
Fungsi uploadLabeledImages() berguna untuk mengupload gambar yang akan diregistrasikan
*/
async function uploadLabeledImages(images, label) {
  try {
    let counter = 0;
    const descriptions = [];
    // Melakukan looping berdasarkan berapa banyak jumlah gambar yang diupload
    for (let i = 0; i < images.length; i++) {
      const img = await canvas.loadImage(images[i]);
      counter = (i / images.length) * 100;
      console.log(`Progress = ${counter}%`);
      // Baca setiap wajah dan simpan deskripsi wajah di array deskripsi
      const detections = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
      descriptions.push(detections.descriptor);
    }

    // Menyimpan data wajah di MongoDB
    const createFace = new FaceModel({
      label: label,
      descriptions: descriptions,
    });
    await createFace.save();
    return true;
  } catch (error) {
    console.log(error);
    return error;
  }
}

/*
Fungsi getDescriptorFromDB() berguna untuk mengecek wajah apakah sudah dikenali oleh sistem
*/
async function getDescriptorFromDB(image) {
  // Dapatkan semua data wajah dari MongoDB dan lewati setiap wajah untuk membaca data
  let faces = await FaceModel.find();

  // Buat objek LabeledFaceDescriptors dari Float32Arrays
  const labeledFaceDescriptors = faces.map((face) => {
    const descriptors = face.descriptions.map((desc) => {
      return new Float32Array(Object.values(desc));
    });
    return new faceapi.LabeledFaceDescriptors(face.label, descriptors);
  });

  // Muat pencocokan wajah untuk menemukan wajah yang cocok dengan ambang batas berat yang diizinkan
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  // Baca gambar menggunakan kanvas
  const img = await canvas.loadImage(image);
  const displaySize = { width: img.width, height: img.height };

  // Mendeteksi satu wajah dalam gambar
  const detections = await faceapi
    .detectSingleFace(img)
    .withFaceLandmarks()
    .withFaceDescriptor();

  // Mengubah ukuran wajah yang terdeteksi
  const resizedDetections = faceapi.resizeResults(detections, displaySize);

  // Temukan kecocokan terbaik untuk wajah yang terdeteksi
  const result = faceMatcher.findBestMatch(resizedDetections.descriptor);

  return result;
}


// Root Endpoint
app.get("/", (_, res) => {
  res.json({
    message: "Face Recognition with Express JS and MongoDB from Face API",
    status: 200,
  });
});

/*
Endpoint untuk mendaftarkan wajah kedalam database
*/
app.post("/recognizing-face", async (req, res) => {
  const File1 = req.files.File1.tempFilePath;
  const File2 = req.files.File2.tempFilePath;
  const File3 = req.files.File3.tempFilePath;
  const label = req.body.label;
  let result = await uploadLabeledImages([File1, File2, File3], label);
  if (result) {
    res.json({ message: "Face data stored successfully" });
  } else {
    res.json({ message: "Something went wrong, please try again." });
  }
});

// Fungsi untuk mnengecek apakah dalam sebuah foto terdapat wajah atau tidak?
async function detectFaces(imagePath) {
  const img = await canvas.loadImage(imagePath);
  const detections = await faceapi.detectSingleFace(img);
  return detections.length > 0;
}

/*
Endpoint untuk mengecek wajah dengan sistem face recognition apakah sudah terdaftar pada database
*/
app.post("/recognizer-face", async (req, res) => {
  const File1 = req.files.File1.tempFilePath;
  let result = await getDescriptorFromDB(File1);
  res.json({ result });
});

/*
Konfigurasi MongoDB
*/
mongoose
  .connect(process.env.MONGO_CONNECTION_STRING, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    useCreateIndex: true,
  })
  .then(() => {
    app.listen(process.env.PORT || 5000);
    console.log("Server is Running and MongoDB is Connected!");
  })
  .catch((err) => {
    console.log(err);
  });
