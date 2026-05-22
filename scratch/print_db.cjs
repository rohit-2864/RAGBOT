const lancedb = require('@lancedb/lancedb');
const path = require('path');

async function main() {
  const dbPath = path.resolve(__dirname, '../data/lancedb');
  const db = await lancedb.connect(dbPath);
  const tables = await db.tableNames();
  console.log('Tables:', tables);
  
  if (tables.includes('chunks')) {
    const table = await db.openTable('chunks');
    const records = await table.query().where('modality = "image"').toArray();
    console.log('Found', records.length, 'image chunks:');
    records.forEach(r => {
      console.log(`ID: ${r.id}, Page: ${r.page}, Source: ${r.source}, Path: ${r.storagePath}, Text: ${r.text}`);
    });
  }
}

main().catch(console.error);
