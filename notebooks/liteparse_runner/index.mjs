
// liteparse_runner/index.mjs
// Minimal LlamaIndex (TypeScript/JS) PDF extraction runner.
// Called from Python via: node index.mjs <pdf_path>
// Writes JSON to stdout: { pages: ["page1 text", ...] }

import { LlamaParseReader } from "llamaindex";
import { readFile } from "fs/promises";

const pdfPath = process.argv[2];
if (!pdfPath) {
  console.error("Usage: node index.mjs <pdf_path>");
  process.exit(1);
}

const reader = new LlamaParseReader({ resultType: "markdown" });
const documents = await reader.loadData(pdfPath);

const output = {
  pages: documents.map((doc) => doc.getText()),
};
process.stdout.write(JSON.stringify(output));
