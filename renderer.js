// renderer.js
const fileInput = document.getElementById('file');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const result = document.getElementById('result');

let arrayBuffer = null;

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  arrayBuffer = null;
  result.textContent = 'Pick an image to begin.';
  predictBtn.disabled = true;
  preview.style.display = 'none';

  if (!f) return;
  if (!/^image\/(jpeg|png)$/i.test(f.type)) {
    alert('Please choose a JPG or PNG image.');
    fileInput.value = '';
    return;
  }

  // Preview
  const reader1 = new FileReader();
  reader1.onload = e => {
    preview.src = e.target.result;
    preview.style.display = 'block';
  };
  reader1.readAsDataURL(f);

  // ArrayBuffer to send to Python
  const reader2 = new FileReader();
  reader2.onload = e => {
    arrayBuffer = e.target.result;  // raw bytes
    predictBtn.disabled = false;
  };
  reader2.readAsArrayBuffer(f);
});

predictBtn.addEventListener('click', async () => {
  if (!arrayBuffer) return;

  result.textContent = 'Predicting...';
  try {
    const out = await window.nativeAPI.predictImage(arrayBuffer);
    // out format: "cat,0.9823"
    const [label, conf] = out.split(',');
    result.textContent = `Prediction: ${label}\nConfidence: ${conf}`;
  } catch (err) {
    result.textContent = 'Error: ' + err.message;
  }
});
