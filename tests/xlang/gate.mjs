// 只照着 spec/tools.json 写的准入组件（JavaScript），不依赖 OpenRSI 的任何代码。
// 逻辑和参考实现 HoldoutGate 的默认配置一致：候选的 holdout 必须严格高于父快照；有 anchor 时不能下降。
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const here = dirname(fileURLToPath(import.meta.url));
const spec = JSON.parse(readFileSync(join(here, "..", "..", "spec", "tools.json"), "utf8"));
const NAMES = ["rsi_component_info", "rsi_gate_check"];
const tools = spec.tools.filter((t) => NAMES.includes(t.name)).map((t) => ({
  name: t.name, description: t.description, inputSchema: t.input_schema, outputSchema: t.output_schema,
}));

const info = () => ({ name: "example/node-gate", version: "0.1.0", protocol: spec.protocol, kinds: ["gate"], tools: NAMES });

function check({ request }) {
  const { candidate_id, baseline, candidate } = request;
  const checks = [];
  const metric = request.primary_metric ?? "holdout";
  if (!(metric in baseline) || !(metric in candidate)) {
    checks.push({ name: "improvement", passed: false, detail: `missing ${metric}` });
  } else {
    checks.push({ name: "improvement", passed: candidate[metric] > baseline[metric],
                  detail: `${metric} ${baseline[metric].toFixed(4)} -> ${candidate[metric].toFixed(4)}` });
  }
  if ("anchor" in baseline && "anchor" in candidate) {
    checks.push({ name: "anchor", passed: candidate.anchor >= baseline.anchor,
                  detail: `anchor ${baseline.anchor} -> ${candidate.anchor}` });
  }
  const failed = checks.filter((c) => !c.passed).map((c) => c.name);
  return { candidate_id, accepted: failed.length === 0, checks,
           reason: failed.length ? `failed: ${failed.join(", ")}` : "all checks passed" };
}

const server = new Server({ name: "example/node-gate", version: "0.1.0" }, { capabilities: { tools: {} } });
server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));
server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const fn = { rsi_component_info: info, rsi_gate_check: check }[req.params.name];
  if (!fn) return { isError: true, content: [{ type: "text", text: `unknown tool ${req.params.name}` }] };
  try {
    const out = fn(req.params.arguments ?? {});
    return { content: [{ type: "text", text: JSON.stringify(out) }], structuredContent: out };
  } catch (e) {
    return { isError: true, content: [{ type: "text", text: String(e) }] };
  }
});
await server.connect(new StdioServerTransport());
