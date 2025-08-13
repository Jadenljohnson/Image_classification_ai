// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('nativeAPI', {
  predictImage: (arrayBuffer) => ipcRenderer.invoke('predict-image', arrayBuffer)
});
