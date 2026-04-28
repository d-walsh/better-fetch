#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import * as cheerio from 'cheerio';
import fetch from 'node-fetch';
import { URL } from 'url';
import TurndownService from 'turndown';
import path from 'path';
import fs from 'fs';
import os from 'os';

// ---------------------------------------------------------------------------
// Structured logging — every crash / timeout gets a forensic trail
// ---------------------------------------------------------------------------

const LOG_DIR = path.join(os.homedir(), '.cache', 'better-fetch');
const LOG_PATH = path.join(LOG_DIR, 'server.log');

try {
  fs.mkdirSync(LOG_DIR, { recursive: true });
} catch {
  // ignore — if we can't create the dir, logging will silently fail
}

function logLine(level: string, msg: string): void {
  const ts = new Date().toISOString();
  const line = `${ts} [${level}] ${msg}\n`;
  try {
    fs.appendFileSync(LOG_PATH, line);
  } catch {
    // never crash on logging failure
  }
}

function logInfo(msg: string): void  { logLine('INFO',  msg); }
function logWarn(msg: string): void  { logLine('WARN',  msg); }
function logError(msg: string): void { logLine('ERROR', msg); }
function logCrit(msg: string): void  { logLine('CRIT',  msg); }

// ---------------------------------------------------------------------------
// Server-level diagnostics (updated by safeCallTool)
// ---------------------------------------------------------------------------

const serverStartTime: number = Date.now();
let toolCallCount: number = 0;
let lastErrorTs: number | null = null;
let lastErrorMsg: string = 'none';

// ---------------------------------------------------------------------------
// Timeout constants (ms) — web fetching is inherently slow
// ---------------------------------------------------------------------------

const TIMEOUTS: Record<string, number> = {
  fetch_website_single:  60_000,   // single-page fetch
  fetch_website_nested: 180_000,   // multi-page crawl
  better_fetch_health_self: 5_000, // health check
};

function getTimeout(toolName: string): number {
  return TIMEOUTS[toolName] ?? 60_000;
}

// ---------------------------------------------------------------------------
// safeCallTool — never-crash wrapper for the entire tool dispatch
// ---------------------------------------------------------------------------

async function safeCallTool(
  toolName: string,
  fn: () => Promise<unknown>
): Promise<{ content: Array<{ type: string; text: string }> }> {
  toolCallCount += 1;
  const callN = toolCallCount;
  const timeoutMs = getTimeout(toolName);

  logInfo(`CALL #${callN} ${toolName}`);
  const t0 = Date.now();

  let timer: ReturnType<typeof setTimeout> | null = null;

  try {
    const result = await Promise.race([
      fn(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`__TIMEOUT__:${timeoutMs}`)),
          timeoutMs
        );
      }),
    ]);
    if (timer !== null) clearTimeout(timer);

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    logInfo(`OK   #${callN} ${toolName} (${elapsed}s)`);

    if (
      result !== null &&
      typeof result === 'object' &&
      'content' in (result as object)
    ) {
      return result as { content: Array<{ type: string; text: string }> };
    }
    return {
      content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
    };
  } catch (err: unknown) {
    if (timer !== null) clearTimeout(timer);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    lastErrorTs = Date.now();

    const isTimeout =
      err instanceof Error && err.message.startsWith('__TIMEOUT__:');

    if (isTimeout) {
      const timeoutS = (timeoutMs / 1000).toFixed(0);
      lastErrorMsg = `timeout after ${elapsed}s`;
      logWarn(`TIMEOUT #${callN} ${toolName} after ${elapsed}s (limit ${timeoutS}s)`);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              error: 'timeout',
              tool: toolName,
              timeout_s: Number(timeoutS),
              message: `${toolName} did not complete within ${timeoutS}s. Check log: ${LOG_PATH}`,
            }),
          },
        ],
      };
    }

    // Everything else — log + return structured error (never crash)
    const errMsg = err instanceof Error ? err.message : String(err);
    const stack  = err instanceof Error ? (err.stack ?? '') : '';
    lastErrorMsg = errMsg;
    logError(`CRASH #${callN} ${toolName} after ${elapsed}s: ${errMsg}\n${stack}`);

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            error: 'exception',
            tool: toolName,
            type: err instanceof Error ? err.constructor.name : typeof err,
            message: errMsg,
          }),
        },
      ],
    };
  }
}

// ---------------------------------------------------------------------------
// Uncaught-exception hooks — forensic logging before process dies
// ---------------------------------------------------------------------------

process.on('uncaughtException', (err: Error) => {
  logCrit(`UNCAUGHT EXCEPTION: ${err.name}: ${err.message}\n${err.stack ?? ''}`);
  process.exit(1);
});

process.on('unhandledRejection', (reason: unknown) => {
  const msg =
    reason instanceof Error
      ? `${reason.name}: ${reason.message}\n${reason.stack ?? ''}`
      : String(reason);
  logCrit(`UNHANDLED REJECTION: ${msg}`);
});

// ---------------------------------------------------------------------------
// Heartbeat — logs "alive" every 60s
// ---------------------------------------------------------------------------

setInterval(() => {
  const uptimeSec = Math.round((Date.now() - serverStartTime) / 1000);
  const lastErrStr = lastErrorTs
    ? `${Math.round((Date.now() - lastErrorTs) / 1000)}s ago`
    : 'none';
  logInfo(
    `HEARTBEAT uptime=${uptimeSec}s calls=${toolCallCount} last_error=${lastErrStr}`
  );
}, 60_000).unref();

logInfo(`SERVER START better-fetch v1.0.1`);

// ---------------------------------------------------------------------------
// Web scraper implementation (unchanged)
// ---------------------------------------------------------------------------

interface FetchOptions {
  maxDepth?: number;
  maxPages?: number;
  sameDomainOnly?: boolean;
  excludePatterns?: string[];
  includePatterns?: string[];
  timeout?: number;
}

interface PageContent {
  url: string;
  title: string;
  content: string;
  links: string[];
  depth: number;
}

class AdvancedWebScraper {
  private turndownService: TurndownService;
  private visitedUrls: Set<string> = new Set();
  private baseUrl: string = '';

  constructor() {
    this.turndownService = new TurndownService({
      headingStyle: 'atx',
      codeBlockStyle: 'fenced',
      bulletListMarker: '-',
    });

    // Custom rules for better markdown conversion
    this.turndownService.addRule('removeScripts', {
      filter: ['script', 'style', 'nav', 'header', 'footer', 'aside'],
      replacement: () => ''
    });

    this.turndownService.addRule('cleanCodeBlocks', {
      filter: 'pre',
      replacement: (content, node) => {
        const code = node.textContent || '';
        return '\n```\n' + code + '\n```\n';
      }
    });
  }

  private async fetchWithTimeout(url: string, timeout: number = 10000): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; MCP-WebScraper/1.0)',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
      });
      clearTimeout(timeoutId);
      return response as unknown as Response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  }

  private extractLinks($: cheerio.CheerioAPI, baseUrl: string): string[] {
    const links: string[] = [];

    $('a[href]').each((_, element) => {
      const href = $(element).attr('href');
      if (href) {
        try {
          const absoluteUrl = new URL(href, baseUrl).toString();
          links.push(absoluteUrl);
        } catch (error) {
          // Invalid URL, skip
        }
      }
    });

    return [...new Set(links)]; // Remove duplicates
  }

  private shouldProcessUrl(url: string, options: FetchOptions): boolean {
    const urlObj = new URL(url);
    const baseUrlObj = new URL(this.baseUrl);

    // Check if same domain only
    if (options.sameDomainOnly && urlObj.hostname !== baseUrlObj.hostname) {
      return false;
    }

    // Check exclude patterns
    if (options.excludePatterns) {
      for (const pattern of options.excludePatterns) {
        if (url.match(new RegExp(pattern, 'i'))) {
          return false;
        }
      }
    }

    // Check include patterns
    if (options.includePatterns && options.includePatterns.length > 0) {
      let matches = false;
      for (const pattern of options.includePatterns) {
        if (url.match(new RegExp(pattern, 'i'))) {
          matches = true;
          break;
        }
      }
      if (!matches) return false;
    }

    // Skip common non-content URLs
    const skipPatterns = [
      /\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|tar|gz)$/i,
      /^mailto:/,
      /^tel:/,
      /^javascript:/,
      /#$/,
      /\/search\?/,
      /\/login/,
      /\/register/,
      /\/cart/,
      /\/checkout/,
    ];

    return !skipPatterns.some(pattern => pattern.test(url));
  }

  private cleanContent($: cheerio.CheerioAPI): string {
    // Remove unwanted elements
    $('script, style, nav, header, footer, aside, .advertisement, .ads, .sidebar, .menu, .navigation').remove();

    // Find main content area
    let contentElement = $('main, article, .content, .main-content, #content, #main').first();

    if (contentElement.length === 0) {
      // Fallback to body if no main content area found
      contentElement = $('body');
    }

    return contentElement.html() || '';
  }

  private generateSectionTitle(url: string, title: string, depth: number): string {
    const urlObj = new URL(url);
    const pathParts = urlObj.pathname.split('/').filter(part => part && part !== 'index.html');

    let sectionTitle = title || pathParts[pathParts.length - 1] || urlObj.hostname;

    // Clean up title
    sectionTitle = sectionTitle
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
      .trim();

    const headerLevel = '#'.repeat(Math.min(depth + 1, 6));
    return `${headerLevel} ${sectionTitle}`;
  }

  private async fetchPageContent(url: string, depth: number, options: FetchOptions): Promise<PageContent | null> {
    if (this.visitedUrls.has(url)) {
      return null;
    }

    this.visitedUrls.add(url);

    try {
      logInfo(`Fetching: ${url} (depth: ${depth})`);

      const response = await this.fetchWithTimeout(url, options.timeout);

      if (!response.ok) {
        logWarn(`Failed to fetch ${url}: ${response.status}`);
        return null;
      }

      const html = await response.text();
      const $ = cheerio.load(html);

      // Extract title
      const title = $('title').text().trim() ||
                   $('h1').first().text().trim() ||
                   'Untitled Page';

      // Clean and extract content
      const cleanHtml = this.cleanContent($);
      const markdownContent = this.turndownService.turndown(cleanHtml);

      // Extract links for potential further processing
      const links = this.extractLinks($, url);

      return {
        url,
        title,
        content: markdownContent,
        links: links.filter(link => this.shouldProcessUrl(link, options)),
        depth
      };

    } catch (error) {
      logError(`Error fetching ${url}: ${error}`);
      return null;
    }
  }

  async scrapeWebsite(startUrl: string, options: FetchOptions = {}): Promise<string> {
    const {
      maxDepth = 2,
      maxPages = 50,
      sameDomainOnly = true,
      timeout = 10000
    } = options;

    this.baseUrl = startUrl;
    this.visitedUrls.clear();

    const allContent: PageContent[] = [];
    const urlsToProcess: Array<{ url: string; depth: number }> = [{ url: startUrl, depth: 0 }];

    while (urlsToProcess.length > 0 && allContent.length < maxPages) {
      const { url, depth } = urlsToProcess.shift()!;

      if (depth > maxDepth || this.visitedUrls.has(url)) {
        continue;
      }

      const pageContent = await this.fetchPageContent(url, depth, options);

      if (pageContent) {
        allContent.push(pageContent);

        // Add child URLs for processing
        if (depth < maxDepth) {
          for (const link of pageContent.links) {
            if (!this.visitedUrls.has(link)) {
              urlsToProcess.push({ url: link, depth: depth + 1 });
            }
          }
        }
      }

      // Small delay to be respectful
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    return this.formatAsMarkdown(allContent, startUrl);
  }

  private formatAsMarkdown(contents: PageContent[], startUrl: string): string {
    const urlObj = new URL(startUrl);
    const siteName = urlObj.hostname;

    let markdown = `# ${siteName} Documentation\n\n`;
    markdown += `*Scraped from: ${startUrl}*\n`;
    markdown += `*Generated on: ${new Date().toISOString()}*\n\n`;

    // Table of contents
    markdown += `## Table of Contents\n\n`;
    contents.forEach((content, _index) => {
      const indent = '  '.repeat(content.depth);
      markdown += `${indent}- [${content.title}](#${this.slugify(content.title)})\n`;
    });
    markdown += '\n---\n\n';

    // Content sections
    contents.forEach(content => {
      const sectionTitle = this.generateSectionTitle(content.url, content.title, content.depth);
      markdown += `${sectionTitle}\n\n`;
      markdown += `*Source: [${content.url}](${content.url})*\n\n`;
      markdown += content.content;
      markdown += '\n\n---\n\n';
    });

    return markdown;
  }

  private slugify(text: string): string {
    return text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/[\s_-]+/g, '-')
      .replace(/^-+|-+$/g, '');
  }
}

// ---------------------------------------------------------------------------
// Tool definitions (signatures and descriptions unchanged)
// ---------------------------------------------------------------------------

const TOOLS: Tool[] = [
  {
    name: "fetch_website_nested",
    description: "Fetch website content with nested URL crawling and convert to clean markdown",
    inputSchema: {
      type: "object",
      properties: {
        url: {
          type: "string",
          description: "The starting URL to fetch and crawl",
        },
        maxDepth: {
          type: "number",
          description: "Maximum depth to crawl (default: 2)",
          default: 2,
        },
        maxPages: {
          type: "number",
          description: "Maximum number of pages to fetch (default: 50)",
          default: 50,
        },
        sameDomainOnly: {
          type: "boolean",
          description: "Only crawl URLs from the same domain (default: true)",
          default: true,
        },
        excludePatterns: {
          type: "array",
          items: { type: "string" },
          description: "Regex patterns for URLs to exclude",
        },
        includePatterns: {
          type: "array",
          items: { type: "string" },
          description: "Regex patterns for URLs to include (if specified, only matching URLs will be processed)",
        },
        timeout: {
          type: "number",
          description: "Request timeout in milliseconds (default: 10000)",
          default: 10000,
        },
      },
      required: ["url"],
    },
  },
  {
    name: "fetch_website_single",
    description: "Fetch content from a single webpage and convert to clean markdown",
    inputSchema: {
      type: "object",
      properties: {
        url: {
          type: "string",
          description: "The URL to fetch",
        },
        timeout: {
          type: "number",
          description: "Request timeout in milliseconds (default: 10000)",
          default: 10000,
        },
      },
      required: ["url"],
    },
  },
  {
    name: "better_fetch_health_self",
    description: "Health check for the better-fetch MCP server. Returns uptime, log path, last error, and total tool call count.",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
];

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

const server = new Server(
  {
    name: "better-fetch",
    version: "1.0.1",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

const scraper = new AdvancedWebScraper();

// Handle tool listing
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

// Handle tool execution — all calls go through safeCallTool
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  return safeCallTool(name, async () => {
    switch (name) {
      case "fetch_website_nested": {
        const {
          url,
          maxDepth = 2,
          maxPages = 50,
          sameDomainOnly = true,
          excludePatterns = [],
          includePatterns = [],
          timeout = 10000,
        } = args as any;

        if (!url) {
          throw new Error("URL is required");
        }

        const options: FetchOptions = {
          maxDepth,
          maxPages,
          sameDomainOnly,
          excludePatterns,
          includePatterns,
          timeout,
        };

        const markdown = await scraper.scrapeWebsite(url, options);
        return { content: [{ type: "text", text: markdown }] };
      }

      case "fetch_website_single": {
        const { url, timeout = 10000 } = args as any;

        if (!url) {
          throw new Error("URL is required");
        }

        const options: FetchOptions = {
          maxDepth: 0,
          maxPages: 1,
          timeout,
        };

        const markdown = await scraper.scrapeWebsite(url, options);
        return { content: [{ type: "text", text: markdown }] };
      }

      case "better_fetch_health_self": {
        const uptimeSec = Math.round((Date.now() - serverStartTime) / 1000);
        const lastErrStr = lastErrorTs
          ? `${Math.round((Date.now() - lastErrorTs) / 1000)}s ago: ${lastErrorMsg}`
          : 'none';

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify({
                status: 'ok',
                uptime_s: uptimeSec,
                total_calls: toolCallCount,
                last_error: lastErrStr,
                log_path: LOG_PATH,
                server: 'better-fetch v1.0.1',
              }, null, 2),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  });
});

// Server-level error handler
server.onerror = (error) => {
  logError(`[MCP Server Error] ${error}`);
};

process.on("SIGINT", async () => {
  logInfo('SIGINT received — shutting down');
  await server.close();
  process.exit(0);
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  logInfo("better-fetch MCP server running on stdio");
}

main().catch((error) => {
  logCrit(`Server failed to start: ${error}`);
  process.exit(1);
});
