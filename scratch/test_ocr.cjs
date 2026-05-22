const fs = require('fs');
const path = require('path');

async function testOCR() {
  const imagePath = process.argv[2];
  if (!imagePath) {
    console.error('Please provide an image path');
    return;
  }

  const formData = new FormData();
  const buffer = fs.readFileSync(imagePath);
  const blob = new Blob([buffer], { type: 'image/png' });
  formData.append('file', blob, path.basename(imagePath));

  try {
    const res = await fetch('http://localhost:3001/api/extract', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    console.log('OCR Result:', data.text);
  } catch (e) {
    console.error('OCR Test Failed:', e);
  }
}

testOCR();
