import React, { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";

/* ================================
   Types
================================ */

interface LegendItem {
  name: string;
  color: [number, number, number, number];
}

interface BarData {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  segments: number[];
  sourceFile?: string;
}

interface ChartDefinition {
  legend: LegendItem[];
  bars: BarData[];
}

/* ================================
   Shaders
================================ */

const VS_SOURCE = `
attribute vec2 a_position;
attribute vec4 a_color;

uniform vec2 u_resolution;
uniform vec2 u_translation;
uniform float u_scale;

varying vec4 v_color;
void main() {
  vec2 position = (a_position + u_translation) * u_scale;

  vec2 zeroToOne = (position + u_resolution / 2.0) / u_resolution;
  vec2 clipSpace = zeroToOne * 2.0 - 1.0;

  gl_Position = vec4(clipSpace * vec2(1.0, -1.0), 0.0, 1.0);
  v_color = a_color;
}
`;

const FS_SOURCE = `
precision mediump float;
varying vec4 v_color;
void main() {
  gl_FragColor = v_color;
}
`;

/* ================================
   Component
================================ */

const BarChart: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const bufferRef = useRef<WebGLBuffer | null>(null);
  const vertexCountRef = useRef(0);
  const borderVertexCountRef = useRef(0);

  const borderBufferRef = useRef<WebGLBuffer | null>(null);

  const [bars, setBars] = useState<BarData[]>([]);
  const [legend, setLegend] = useState<LegendItem[]>([]);
  const [scale, setScale] = useState(1);
  const [translation, setTranslation] = useState({ x: 0, y: 0 });
  const [showLegend, setShowLegend] = useState(false);
  const [uiVisible, setUiVisible] = useState(true);
  const [culling, setCulling] = useState(true);
  const [showShortcuts, setShowShortcuts] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const indexBufferRef = useRef<WebGLBuffer | null>(null);
  const indexCountRef = useRef<number>(0);
  const baseQuadsRef = useRef<Quad[]>([]);
  const quadtreeRef = useRef<QuadNode | null>(null);
  const [fileLabels, setFileLabels] = useState<{name: string, x: number, y: number}[]>([]);

  /* ================================
     File Upload
  ================================ */

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    const allBars: BarData[] = [];
    const labels: {name: string, x:number, y: number}[] = [];
    let combinedLegend: LegendItem[] = [];
    const verticalSpacing = 300; // Offset between datasets
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const text = await file.text();
      
      try {
        const parsed: ChartDefinition = JSON.parse(text);
        const currentYOffset = -(i * verticalSpacing);
        
        const minX = Math.min(...parsed.bars.map(b => b.x));
        // Offset the Y position of bars for each subsequent file
        const offsetBars = parsed.bars.map(bar => ({
          ...bar,
          y: bar.y + currentYOffset,
          sourceFile: file.name
        }));

        allBars.push(...offsetBars);
        labels.push({ name: file.name, x: minX, y: currentYOffset });
        
        // Merge legends if they are different (or just take the first)
        if (i === 0) combinedLegend = parsed.legend;
        
      } catch (err) {
        console.error(`Failed to parse ${file.name}`, err);
      }
    }

    const globalMinX = allBars.length > 0 ? Math.min(...allBars.map(b => b.x)) : 0;


    setBars(allBars);
    setLegend(combinedLegend);
    setFileLabels(labels);
    setScale(1);
    setTranslation({ x: -globalMinX + 50, y: 0 });
  };

  /* ================================
     WebGL Initialization (once)
  ================================ */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2");
    if (!gl) return;

    glRef.current = gl;

    const createShader = (
      gl: WebGLRenderingContext,
      type: number,
      source: string
    ) => {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return shader;
    };

    const program = gl.createProgram()!;
    gl.attachShader(program, createShader(gl, gl.VERTEX_SHADER, VS_SOURCE));
    gl.attachShader(program, createShader(gl, gl.FRAGMENT_SHADER, FS_SOURCE));
    gl.linkProgram(program);

    programRef.current = program;
    bufferRef.current = gl.createBuffer();
  }, []);

  // Types
  type Quad = { x: number; y: number; w: number; h: number; color?: [number,number,number,number] };
  type Rect = { x: number; y: number; w: number; h: number };

  // Utility : AABB intersection / containment
  function rectIntersects(a: Rect, b: Rect) {
    return !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
  }
  function rectContains(a: Rect, b: Rect) {
    // does A fully contain B?
    return a.x <= b.x && a.y <= b.y && a.x + a.w >= b.x + b.w && a.y + a.h >= b.y + b.h;
  }

  // Quadtree node
  class QuadNode {
    bounds: Rect;
    indices: number[] = [];      // indices of base quads that are stored in this node
    children: QuadNode[] | null = null;
    depth: number;

    constructor(bounds: Rect, depth = 0) {
      this.bounds = bounds;
      this.depth = depth;
    }
  }

  // Build quadtree from base quads array
  function buildQuadTree(baseQuads: Quad[], options?: {
    maxDepth?: number; capacity?: number; rootBounds?: Rect;
  }) {
    const maxDepth = options?.maxDepth ?? 10;
    const capacity = options?.capacity ?? 8;

    // compute root bounds if not provided
    let rootBounds = options?.rootBounds;
    if (!rootBounds) {
      if (baseQuads.length === 0) rootBounds = { x: 0, y: 0, w: 0, h: 0 };
      else {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        baseQuads.forEach(q => {
          minX = Math.min(minX, q.x);
          minY = Math.min(minY, q.y);
          maxX = Math.max(maxX, q.x + q.w);
          maxY = Math.max(maxY, q.y + q.h);
        });
        rootBounds = { x: minX, y: minY, w: Math.max(1e-6, maxX - minX), h: Math.max(1e-6, maxY - minY) };
      }
    }

    const root = new QuadNode(rootBounds, 0);

    function subdivide(node: QuadNode) {
      const { x, y, w, h } = node.bounds;
      const hw = w / 2;
      const hh = h / 2;
      node.children = [
        new QuadNode({ x: x,     y: y,     w: hw, h: hh }, node.depth + 1), // top-left
        new QuadNode({ x: x+hw,  y: y,     w: hw, h: hh }, node.depth + 1), // top-right
        new QuadNode({ x: x,     y: y+hh,  w: hw, h: hh }, node.depth + 1), // bottom-left
        new QuadNode({ x: x+hw,  y: y+hh,  w: hw, h: hh }, node.depth + 1), // bottom-right
      ];
    }

    function tryInsert(node: QuadNode, quadIndex: number) {
      const quad = baseQuads[quadIndex];
      const qRect = { x: quad.x, y: quad.y, w: quad.w, h: quad.h };

      // If node has children, attempt to push it down into a single child that fully contains it.
      if (node.children) {
        for (const child of node.children) {
          if (rectContains(child.bounds, qRect)) {
            tryInsert(child, quadIndex);
            return;
          }
        }
        // otherwise it does not fully fit into any single child -> keep in current node
        node.indices.push(quadIndex);
        return;
      }

      // if leaf: store
      node.indices.push(quadIndex);

      // split if capacity exceeded and not at max depth
      if (node.indices.length > capacity && node.depth < maxDepth) {
        subdivide(node);
        // re-distribute
        const toReinsert = node.indices.slice();
        node.indices.length = 0;
        toReinsert.forEach(idx => tryInsert(node, idx));
      }
    }

    // Insert all quads
    for (let i = 0; i < baseQuads.length; i++) tryInsert(root, i);

    return root;
  }

  // Query: returns indices of baseQuads (only from leaf nodes) that intersect viewRect
  function queryVisibleIndices(root: QuadNode, baseQuads: Quad[], viewRect: Rect) {
    const out: number[] = [];

    function visit(node: QuadNode) {
      if (!rectIntersects(node.bounds, viewRect)) return; // node fully outside

      // If node fully inside view, we can gather all indices in this subtree quickly:
      if (rectContains(viewRect, node.bounds)) {
        // collect all indices from leaves under this node without per-quad intersection tests
        collectAllLeafIndices(node);
        return;
      }

      // Partial intersection -> either dive into children or check leaf indices individually
      if (node.children) {
        node.children.forEach(visit);
      } 
      if(node.indices) {
        // node is leaf: check each stored quad against the view
        for (const idx of node.indices) {
          const q = baseQuads[idx];
          if (rectIntersects(viewRect, { x: q.x, y: q.y, w: q.w, h: q.h })) out.push(idx);
        }
      }

      }

    function collectAllLeafIndices(node: QuadNode) {
      if (node.children) {
        node.children.forEach(collectAllLeafIndices);
      } else {
        // leaf: add every index (no per-quad test)
        out.push(...node.indices);
      }
    }

    visit(root);
    // Remove duplicates (just in case) and return
    return Array.from(new Set(out));
  }

  // Compute view rectangle in *world* coordinates from uniforms used in vertex shader
  // Shader does: position' = (pos + u_translation) * u_scale
  // then position' must be in [-u_resolution/2, u_resolution/2] to be inside the viewport
  function computeViewRect(u_resolution: {x:number,y:number}, u_translation: {x:number,y:number}, u_scale: number) {
    const halfW = u_resolution.x / (2 * u_scale);
    const halfH = u_resolution.y / (2 * u_scale);

    const left = -halfW - u_translation.x;
    const right = halfW - u_translation.x;
    const top = -halfH - u_translation.y;
    const bottom = halfH - u_translation.y;

    return { x: left, y: top, w: right - left, h: bottom - top };
  }

  const updateIndexBufferFromCulling = useCallback((
    width: number, height: number
  ) => {
    const gl = glRef.current;
    const indexBuffer = indexBufferRef.current;
    const baseQuads = baseQuadsRef.current;
    const quadtree = quadtreeRef.current;
    const canvas = canvasRef.current;

    if (!gl || !indexBuffer || !quadtree || !canvas) return;

    const viewRect = computeViewRect(
      { x: width, y: height },
      translation,
      scale
    );

    const visibleQuadIndices = culling ? queryVisibleIndices(
      quadtree,
      baseQuads,
      viewRect
    ) 
    : baseQuads.map((_, i) => i);

    const indexData: number[] = [];

    visibleQuadIndices.forEach((quadIndex) => {
      const baseVertex = quadIndex * 6;

      indexData.push(
        baseVertex + 0,
        baseVertex + 1,
        baseVertex + 2,
        baseVertex + 3,
        baseVertex + 4,
        baseVertex + 5
      );
    });

    const uintData = new Uint32Array(indexData);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, uintData, gl.DYNAMIC_DRAW);

    indexCountRef.current = uintData.length;

  }, [scale, translation]);

  // generate bar graph data
  const generateGeometry = useCallback(() => {
    const gl = glRef.current;
    const buffer = bufferRef.current;
    if (!gl || !buffer) return;

    const vertexData: number[] = [];
    const baseQuads: Quad[] = [];

    bars.forEach((bar) => {
      const total = bar.segments.reduce((a, b) => a + b, 0);
      if (total === 0) return;

      let accumulatedHeight = 0;

      bar.segments.forEach((value, i) => {
        const segmentHeight = (value / total) * bar.h;

        const x1 = bar.x;
        const x2 = bar.x + bar.w;

        const yTop = bar.y - accumulatedHeight;
        const yBottom = bar.y - accumulatedHeight - segmentHeight;

        const color = legend[i]?.color ?? [1, 1, 1, 1];

        // Store quad for quadtree
        baseQuads.push({
          x: x1,
          y: yBottom,
          w: x2 - x1,
          h: yTop - yBottom,
          color,
        });

        // Push 6 vertices (static order)
        vertexData.push(
          x1, yTop,    ...color,
          x2, yTop,    ...color,
          x1, yBottom, ...color,

          x1, yBottom, ...color,
          x2, yTop,    ...color,
          x2, yBottom, ...color
        );

        accumulatedHeight += segmentHeight;
      });
    });

    baseQuadsRef.current = baseQuads;
    quadtreeRef.current = buildQuadTree(baseQuads);

    const floatData = new Float32Array(vertexData);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, floatData, gl.STATIC_DRAW);

    vertexCountRef.current = floatData.length / 6;

  }, [bars, legend]);

  //
  // generate border buffer
  //
  const generateBorderBuffer = useCallback(() => {
    const gl = glRef.current;
    if (!gl) return;

    const borderBuffer = gl.createBuffer();
    borderBufferRef.current = borderBuffer;

    const cx = 0;
    const cy = 0;

    const borderColor = [0.4, 0.4, 0.4, 1];

    const data = new Float32Array([
      cx - 300, cy + 100, ...borderColor,
      cx + 300, cy + 100, ...borderColor,
      cx + 300, cy - 200, ...borderColor,
      cx - 300, cy - 200, ...borderColor,
    ]);

    gl.bindBuffer(gl.ARRAY_BUFFER, borderBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);

    borderVertexCountRef.current = 4;
  }, []);

  /* ================================
     Draw Function
  ================================ */

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const gl = glRef.current;
    const program = programRef.current;
    const buffer = bufferRef.current;
    const borderBuffer = borderBufferRef.current;

    if (!canvas || !gl || !program || !buffer || !borderBuffer) return;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.05, 0.05, 0.05, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(program);

    const positionLoc = gl.getAttribLocation(program, "a_position");
    const colorLoc = gl.getAttribLocation(program, "a_color");

    const resLoc = gl.getUniformLocation(program, "u_resolution");
    const transLoc = gl.getUniformLocation(program, "u_translation");
    const scaleLoc = gl.getUniformLocation(program, "u_scale");

    gl.uniform2f(resLoc, canvas.width, canvas.height);
    gl.uniform2f(transLoc, translation.x, translation.y);
    gl.uniform1f(scaleLoc, scale);

    const stride = 6 * 4; // 6 floats per vertex

    updateIndexBufferFromCulling(canvas.width, canvas.height);

    /* ===== Draw Bars ===== */
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(
      positionLoc,
      2,
      gl.FLOAT,
      false,
      stride,
      0
    );

    gl.enableVertexAttribArray(colorLoc);
    gl.vertexAttribPointer(
      colorLoc,
      4,
      gl.FLOAT,
      false,
      stride,
      2 * 4
    );

    // ALWAYS bind index buffer right before drawing
  const indexBuffer = indexBufferRef.current;
  if (!indexBuffer) {
    console.log("Index buffer is NULL, exiting draw()");
    return 0;
  }

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.drawElements(
      gl.TRIANGLES,
      indexCountRef.current,
      gl.UNSIGNED_INT,
      0
    );

    /* ===== Draw Border ===== */
    gl.bindBuffer(gl.ARRAY_BUFFER, borderBuffer);

    gl.vertexAttribPointer(
      positionLoc,
      2,
      gl.FLOAT,
      false,
      stride,
      0
    );

    gl.vertexAttribPointer(
      colorLoc,
      4,
      gl.FLOAT,
      false,
      stride,
      2 * 4
    );

    gl.drawArrays(gl.LINE_LOOP, 0, borderVertexCountRef.current);

  }, [scale, translation]);

  useEffect(() => {
    if (bars.length > 0) {
      generateGeometry();
      generateBorderBuffer();
    }
  }, [bars, legend, generateGeometry, generateBorderBuffer]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const gl = glRef.current;
    if (!gl) return;
    indexBufferRef.current = gl.createBuffer();
  }, []);

  /* ================================
     Zoom Handling
  ================================ */

  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();

      const zoomIntensity = 0.001;
      const delta = -e.deltaY * zoomIntensity;
      const newScale = Math.max(0.1, Math.min(scale * (1 + delta), 20));

      const cx = window.innerWidth / 2;
      const cy = window.innerHeight / 2;

      // Zoom in → anchor to mouse
      const anchorX = e.deltaY < 0 ? e.clientX - cx : 0;
      const anchorY = e.deltaY < 0 ? e.clientY - cy : 0;

      const worldX = anchorX / scale - translation.x;
      const worldY = anchorY / scale - translation.y;

      setTranslation({
        x: anchorX / newScale - worldX,
        y: anchorY / newScale - worldY,
      });

      setScale(newScale);
    };

    window.addEventListener("wheel", handleWheel, { passive: false });
    return () => window.removeEventListener("wheel", handleWheel);
  }, [scale, translation]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input field
      const active = document.activeElement;
      if (
        active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // alt+K → toggle UI
      if (e.altKey && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setUiVisible(prev => !prev);
      }

      // alt+M → toggle shortcuts panel
      if (e.altKey && e.key.toLowerCase() === "m") {
        e.preventDefault();
        setShowShortcuts(prev => !prev);
      }

      // alt+c -> toggle culling 
      if(e.altKey && e.key.toLowerCase() === "c") {
        e.preventDefault();
        setCulling(prev => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);


  // mouse dragging feature

  const handleMouseDown = (e: React.MouseEvent) => {
  // Only start dragging if clicking the canvas (not UI buttons)
  if ((e.target as HTMLElement).tagName === "CANVAS") {
    setIsDragging(true);
    setLastMousePos({ x: e.clientX, y: e.clientY });
  }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;

    // Calculate how far the mouse moved since the last frame
    const dx = e.clientX - lastMousePos.x;
    const dy = e.clientY - lastMousePos.y;

    // Update translation. 
    // We divide by scale so that dragging feels consistent regardless of zoom level.
    setTranslation(prev => ({
      x: prev.x + dx / scale,
      y: prev.y + dy / scale
    }));

    // Update the reference point for the next movement
    setLastMousePos({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  /* ================================
     JSX
  ================================ */

  const hasValidData = legend.length > 0 && bars.length > 0;

  return (
    <div 
    className="webgl-container"
    onMouseDown={handleMouseDown}
    onMouseMove={handleMouseMove}
    onMouseUp={handleMouseUp}
    onMouseLeave={handleMouseUp} // Stop dragging if mouse leaves window
    style={{ cursor: isDragging ? 'grabbing' : 'crosshair' }}
    >
      <canvas ref={canvasRef} />
      {uiVisible && (
        <div className="ui-overlay">
          {fileLabels.map((file, i) => {
            const cx = window.innerWidth / 2;
            const cy = window.innerHeight / 2;

            // Calculate screen coordinates based on zoom and pan
            const xPos = cx + (file.x + translation.x) * scale;
            const yPos = cy + (file.y + translation.y) * scale;

            return (
              <div 
                key={`file-${i}`} 
                className="file-header-label" 
                style={{ 
                  left: `${xPos}px`, 
                  top: `${yPos - 100 * scale}px`, // Place 20px above the dataset
                }}
              >
                SOURCE: {file.name}
              </div>
            );
          })}
          {bars.map((bar, i) => {
            // 1. Zoom Threshold: Don't show labels if we are zoomed too far out
            if (scale < 0.8) return null;

            const cx = window.innerWidth / 2;
            const cy = window.innerHeight / 2;

            // Calculate screen position
            const left = cx + (bar.x + bar.w / 2 + translation.x) * scale;
            const top = cy + (bar.y + 20 + translation.y) * scale;

            // 2. Viewport Culling: Only render if the label is actually on screen
            const padding = 100; // Buffer to prevent labels from popping in at the edges
            if (
              left < -padding || 
              left > window.innerWidth + padding || 
              top < -padding || 
              top > window.innerHeight + padding
            ) {
              return null;
            }

            return (
              <div key={i} className="label" style={{ left, top }}>
                {bar.label}
              </div>
            );
          })} 
        </div>
      )}
      {uiVisible && (
        <div className="controls">
          <input type="file" accept=".json" multiple onChange={handleFileUpload} />
          <button
            onClick={() => {
              setShowLegend(prev => !prev);
            }}
            disabled={!hasValidData}
            className={!hasValidData ? "disabled-button" : ""}
          >
            {showLegend ? "Hide Legend" : "Show Legend"}
          </button>

          <div className="zoom-indicator">
            ZOOM: {scale.toFixed(2)}x
          </div>
          <div className="culling-indicator">
            CULLING: {culling ? "ON" : "OFF"}
          </div>

          <button 
            onClick={() => {
              const minX = bars.length > 0 ? Math.min(...bars.map(b => b.x)) : 0;
              setScale(1);
              setTranslation({ x: -minX + 50, y: 0 });
            }}
          >
            Reset View
          </button>

          {/* Legend */}
            <div className="legend"
            style={{ display: showLegend ? "flex" : "none" }}
            >
              {legend.map((item, i) => {
                const r = Math.round(item.color[0] * 255);
                const g = Math.round(item.color[1] * 255);
                const b = Math.round(item.color[2] * 255);
                const a = item.color[3];

                return (
                  <div key={i} className="legend-item">
                    <div
                      className="legend-color"
                      style={{
                        backgroundColor: `rgba(${r}, ${g}, ${b}, ${a})`
                      }}
                    />
                    <span>{item.name}</span>
                  </div>
                );
              })}
            </div>
        </div>
      )}
      {showShortcuts && (
        <div className="shortcuts-panel">
          <div className="shortcuts-title">Keyboard Shortcuts</div>
          <div>alt + K → Toggle UI</div>
          <div>alt + M → Toggle Help</div>
          <div>alt + c → Toggle culling</div>
          <div>Mouse Wheel → Zoom</div>
        </div>
      )}
    </div>
  );
};

export default BarChart;