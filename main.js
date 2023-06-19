import './style.css';
import 'webrtc-adapter';


import firebase from 'firebase/app';
import 'firebase/firestore';

const firebaseConfig = {
  // your config
  apiKey: "AIzaSyA3xO6vGVRARsbyN6Ygy7pRrRL9jZR2aOA",
  authDomain: "video-chat-1f4d1.firebaseapp.com",
  projectId: "video-chat-1f4d1",
  storageBucket: "video-chat-1f4d1.appspot.com",
  messagingSenderId: "169066770490",
  appId: "1:169066770490:web:8107c54a4ffddb1e644503",
  measurementId: "G-0VWZWVPMMZ"
};

if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
}
const firestore = firebase.firestore();

const servers = {
  iceServers: [
    {
      urls: ['stun:stun1.l.google.com:19302', 'stun:stun2.l.google.com:19302'],
    },
  ],
  iceCandidatePoolSize: 10,
};

// Global State
const pc = new RTCPeerConnection(servers);
let localStream = null;
let remoteStream = null;

// HTML elements
const webcamButton = document.getElementById('webcamButton');
const webcamVideo = document.getElementById('webcamVideo');
const callButton = document.getElementById('callButton');
const callInput = document.getElementById('callInput');
const answerButton = document.getElementById('answerButton');
const remoteVideo = document.getElementById('remoteVideo');
const hangupButton = document.getElementById('hangupButton');

// 1. Setup media sources

webcamButton.onclick = async () => {
  localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
  remoteStream = new MediaStream();

  // Push tracks from local stream to peer connection
  localStream.getTracks().forEach((track) => {
    pc.addTrack(track, localStream);
  });

  // Pull tracks from remote stream, add to video stream
  pc.ontrack = (event) => {
    event.streams[0].getTracks().forEach((track) => {
      remoteStream.addTrack(track);
    });
  };

  webcamVideo.srcObject = localStream;
  remoteVideo.srcObject = remoteStream;

  callButton.disabled = false;
  answerButton.disabled = false;
  webcamButton.disabled = true;
};

// 2. Create an offer
callButton.onclick = async () => {
  // Reference Firestore collections for signaling
  const callDoc = firestore.collection('calls').doc();
  const offerCandidates = callDoc.collection('offerCandidates');
  const answerCandidates = callDoc.collection('answerCandidates');

  callInput.value = callDoc.id;

  // Get candidates for caller, save to db
  pc.onicecandidate = (event) => {
    event.candidate && offerCandidates.add(event.candidate.toJSON());
  };

  // Create offer
  const offerDescription = await pc.createOffer();
  await pc.setLocalDescription(offerDescription);

  const offer = {
    sdp: offerDescription.sdp,
    type: offerDescription.type,
  };

  await callDoc.set({ offer });

  // Listen for remote answer
  callDoc.onSnapshot((snapshot) => {
    const data = snapshot.data();
    if (!pc.currentRemoteDescription && data?.answer) {
      const answerDescription = new RTCSessionDescription(data.answer);
      pc.setRemoteDescription(answerDescription);
    }
  });

  // When answered, add candidate to peer connection
  answerCandidates.onSnapshot((snapshot) => {
    snapshot.docChanges().forEach((change) => {
      if (change.type === 'added') {
        const candidate = new RTCIceCandidate(change.doc.data());
        pc.addIceCandidate(candidate);
      }
    });
  });

  hangupButton.disabled = false;
};

// 3. Answer the call with the unique ID
answerButton.onclick = async () => {
  const callId = callInput.value;
  const callDoc = firestore.collection('calls').doc(callId);
  const answerCandidates = callDoc.collection('answerCandidates');
  const offerCandidates = callDoc.collection('offerCandidates');

  pc.onicecandidate = (event) => {
    event.candidate && answerCandidates.add(event.candidate.toJSON());
  };

  const callData = (await callDoc.get()).data();

  const offerDescription = callData.offer;
  await pc.setRemoteDescription(new RTCSessionDescription(offerDescription));

  const answerDescription = await pc.createAnswer();
  await pc.setLocalDescription(answerDescription);

  const answer = {
    type: answerDescription.type,
    sdp: answerDescription.sdp,
  };

  await callDoc.update({ answer });

  offerCandidates.onSnapshot((snapshot) => {
    snapshot.docChanges().forEach((change) => {
      console.log(change);
      if (change.type === 'added') {
        let data = change.doc.data();
        pc.addIceCandidate(new RTCIceCandidate(data));
      }
    });
  });
};

// Load the TensorFlow.js library
import * as tf from '@tensorflow/tfjs';

// Load the Teachable Machine model
const URL = "https://teachablemachine.withgoogle.com/models/AA7CGSTNt/";
// const URL = "https://teachablemachine.withgoogle.com/models/cqbb-nRkh/";
const modelUrl = URL + "model.json";
const metadataURL = URL + "metadata.json";

async function loadModel() {
  const model = await tf.loadLayersModel(modelUrl);
  console.log('Model Loaded');

  const metadata = await fetch(metadataURL).then((res) => res.json());
  console.log(metadata);

  navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => {
      const videoElement = document.getElementById('remoteVideo');
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
        const output = await model.predict(batched);
        const predictions = await output.data();
        console.log(predictions);

        // const labelIndex = tf.argMax(output, { axis: 1 }).dataSync()[0];
        const labelIndex = tf.argMax(output, 1).dataSync()[0];
        const label = metadata.labels[labelIndex];
        console.log(label);

        let translatedText = document.getElementById('translatedText').innerHTML = label;

        // Cleanup
        tensor.dispose();
        offset.dispose();
        normalized.dispose();
        batched.dispose();
        tf.dispose([output, predictions]);
      }, 1000 / 30);
    });
}

loadModel();
