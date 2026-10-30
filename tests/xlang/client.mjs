// JavaScript 客户端调用 Python 写的判别器：node client.mjs <启动判别器的命令...>
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const [command, ...args] = process.argv.slice(2);
const transport = new StdioClientTransport({ command, args, env: { ...process.env } });
const client = new Client({ name: "openrsi-xlang-client", version: "0.1.0" });
await client.connect(transport);

const call = async (name, args) => {
  const r = await client.callTool({ name, arguments: args });
  if (r.isError) throw new Error(`${name}: ${r.content.map((c) => c.text).join(" ")}`);
  return r.structuredContent;
};

const info = await call("rsi_component_info", {});
const task = { id: "t1", input: "capital of France?", split: "train", reference: { expected: "Paris" } };
const traj = (output) => ({ id: `tr-${output}`, task_id: "t1", output, steps: [] });
const judge = async (output) => {
  const evidence = await call("rsi_evidence_collect", { task, trajectory: traj(output) });
  const list = Array.isArray(evidence) ? evidence : evidence.result ?? evidence.evidence ?? [];
  return call("rsi_judge_evaluate", { task, trajectory: traj(output), evidence: list });
};
const right = await judge("Paris");
const wrong = await judge("Lyon");
console.log(JSON.stringify({ kinds: info.kinds, right: { passed: right.passed, score: right.score },
                             wrong: { passed: wrong.passed, score: wrong.score } }));
await client.close();
