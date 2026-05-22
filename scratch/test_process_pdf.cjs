async function test() {
  try {
    const res = await fetch('http://localhost:3000/api/documents/process-pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        storagePath: 'cb737475-552c-4234-96e1-061deebdd49a.pdf',
        originalName: 'DL1.pdf'
      })
    });
    console.log('Status:', res.status);
    const data = await res.json();
    console.log('Count:', data.count || 0);
    console.log('Status Details:', data.status || 'no status');
    if (data.error) {
      console.log('Error Details:', data.error);
    }
  } catch (err) {
    console.error('Fetch error:', err);
  }
}

test();
