// config.ts — Parse S-expression configuration file.
// Format: (config (key value) (key value) ...)

import { tokenize } from "./lexer";

export function parseConfig(text: string): Map<string, f64> {
  const config = new Map<string, f64>();
  const tokens = tokenize(text);
  let i = 0;
  // Skip opening ( and "config"
  if (i < tokens.length && tokens[i] == "(") i++;
  if (i < tokens.length && tokens[i] == "config") i++;
  while (i < tokens.length) {
    if (tokens[i] == ")") break;
    if (tokens[i] == "(") {
      i++;
      if (i + 1 < tokens.length && tokens[i + 1] != ")") {
        const key = tokens[i];
        i++;
        const value = parseFloat(tokens[i]);
        i++;
        config.set(key, value);
      }
      if (i < tokens.length && tokens[i] == ")") i++;
    } else {
      i++;
    }
  }
  return config;
}

export function configI32(config: Map<string, f64>, key: string, fallback: i32): i32 {
  if (config.has(key)) return i32(config.get(key));
  return fallback;
}

export function configF32(config: Map<string, f64>, key: string, fallback: f32): f32 {
  if (config.has(key)) return f32(config.get(key));
  return fallback;
}
