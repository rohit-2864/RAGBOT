const fs = require('fs');
const path = require('path');

async function testEmbed() {
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
    const res = await fetch('http://localhost:3001/api/embed/image', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
        console.error('Embed Test Failed:', await res.text());
        return;
    }
    const data = await res.json();
    console.log('Embed Result:', data.embedding ? 'Success (Vector received)' : 'Failed');
    console.log('Dimensions:', data.dimensions);
  } catch (e) {
    console.error('Embed Test Error:', e);
  }
}

testEmbed();
