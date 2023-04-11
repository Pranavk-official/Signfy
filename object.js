// Load the TensorFlow.js library
import * as tf from '@tensorflow/tfjs';

// Load the Teachable Machine model
const URL = "https://teachablemachine.withgoogle.com/models/cqbb-nRkh/";
const modelUrl = URL + "model.json";
const metadataURL = URL + "metadata.json";

async function loadModel() {
  const model = await tf.loadLayersModel(modelUrl, metadataURL);
  console.log('Model Loaded');
  
  navigator.mediaDevices.getUserMedia({ video: true, audio: false })
  .then((stream) => {
    const videoElement = document.getElementById('webcamVideo');
    videoElement.srcObject = stream;
    videoElement.play();

    setInterval(async () => {
      const canvasElement = document.createElement('canvas');
      const canvasContext = canvasElement.getContext('2d');
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
      canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
      const image = canvasContext.getImageData(0, 0, canvasElement.width, canvasElement.height);
      
      // Resize the image to the expected size of the model
      const tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([224, 224]).toFloat();
      const offset = tf.scalar(127.5);
      const normalized = tensor.sub(offset).div(offset);
      const batched = normalized.reshape([1, 224, 224, 3]);
      
      // Predict the class probabilities using the model
      const output = await model.predict(batched).data();
      console.log(output);
      
      // Cleanup
      tensor.dispose();
      offset.dispose();
      normalized.dispose();
      batched.dispose();
    }, 1000 / 30);
  });
}

loadModel();
