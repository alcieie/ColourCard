/* =================================================================
   1. READING THE IMAGE
   The image is drawn into an offscreen canvas at a small size.
   Downsampling is the whole trick for speed: a 4000px photo has
   16 million pixels, but ~40,000 of them describe its colour
   distribution just as well.
   ================================================================= */
const MAX_DIM = 200;

/* -----------------------------------------------------------------
   sRGB <-> CIELAB
   Euclidean distance in Lab is dE*ab, which tracks perceived
   difference far better than distance in RGB does.
   ----------------------------------------------------------------- */
const Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;   // D65 white point
const D3 = 0.008856451679, SLOPE = 7.787037037, OFFSET = 4 / 29;

// Undoing the sRGB transfer function is a pow() per channel per pixel,
// but there are only 256 possible inputs. Precompute all of them.
const LINEAR = new Float64Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

const f = t => (t > D3 ? Math.cbrt(t) : SLOPE * t + OFFSET);
const fInv = t => (t * t * t > D3 ? t * t * t : (t - OFFSET) / SLOPE);

function rgbToLab(r, g, b) {
  const R = LINEAR[r], G = LINEAR[g], B = LINEAR[b];
  const X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B;
  const Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B;
  const Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B;
  const fx = f(X / Xn), fy = f(Y / Yn), fz = f(Z / Zn);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function labToRgb(L, a, bb) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - bb / 200;
  const X = Xn * fInv(fx), Y = Yn * fInv(fy), Z = Zn * fInv(fz);
  const R =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  const G = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  const B =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  // Lab is larger than the sRGB gamut, so clamping is required —
  // some centroids land on colours this display cannot make.
  const enc = v => {
    v = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
    return Math.max(0, Math.min(255, Math.round(v * 255)));
  };
  return [enc(R), enc(G), enc(B)];
}

function pixelsFrom(img) {
  const scale = Math.min(1, MAX_DIM / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));

  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);

  const data = ctx.getImageData(0, 0, w, h).data;
  const out = [];
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 125) continue;            // skip transparent pixels
    out.push([data[i], data[i + 1], data[i + 2]]);
  }
  return out;
}

/* =================================================================
   2. K-MEANS CLUSTERING
   Group the pixels into k clusters in RGB space. Each cluster's
   mean is a palette colour; its size is that colour's share.
   ================================================================= */
function dist2(a, b) {
  const dr = a[0] - b[0], dg = a[1] - b[1], db = a[2] - b[2];
  return dr * dr + dg * dg + db * db;
}

// k-means++ seeding: pick starting points that are spread out, so
// the result doesn't depend on a lucky random draw.
function seed(pixels, k) {
  const cents = [pixels[(Math.random() * pixels.length) | 0].slice()];
  while (cents.length < k) {
    let total = 0;
    const d2 = new Float64Array(pixels.length);
    for (let i = 0; i < pixels.length; i++) {
      let best = Infinity;
      for (const c of cents) {
        const d = dist2(pixels[i], c);
        if (d < best) best = d;
      }
      d2[i] = best;
      total += best;
    }
    if (total === 0) break;                      // image is one flat colour
    let r = Math.random() * total, idx = 0;
    for (let i = 0; i < d2.length; i++) { r -= d2[i]; if (r <= 0) { idx = i; break; } }
    cents.push(pixels[idx].slice());
  }
  return cents;
}

function kmeans(pixels, k, iters = 24) {
  let cents = seed(pixels, k);
  k = cents.length;
  const assign = new Int32Array(pixels.length).fill(-1);

  for (let it = 0; it < iters; it++) {
    let moved = false;

    // assignment step
    for (let i = 0; i < pixels.length; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const d = dist2(pixels[i], cents[c]);
        if (d < bd) { bd = d; best = c; }
      }
      if (assign[i] !== best) { assign[i] = best; moved = true; }
    }

    // update step
    const sums = Array.from({ length: k }, () => [0, 0, 0, 0]);
    for (let i = 0; i < pixels.length; i++) {
      const s = sums[assign[i]], p = pixels[i];
      s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; s[3]++;
    }
    for (let c = 0; c < k; c++) {
      if (sums[c][3] > 0) cents[c] = [sums[c][0] / sums[c][3], sums[c][1] / sums[c][3], sums[c][2] / sums[c][3]];
    }

    if (!moved) break;                           // converged
  }

  const counts = new Array(k).fill(0);
  for (let i = 0; i < pixels.length; i++) counts[assign[i]]++;

  // Centroids come back in whatever space the pixels were in.
  // The caller converts them to RGB for display.
  return cents
    .map((c, i) => ({ c, share: counts[i] / pixels.length }))
    .filter(x => x.share > 0)
    .sort((a, b) => b.share - a.share);
}

/* =================================================================
   3. DRAWING THE STRIP
   ================================================================= */
const hex = c => '#' + [c.r, c.g, c.b].map(v => v.toString(16).padStart(2, '0')).join('').toUpperCase();

// WCAG relative luminance, used to decide black or white label text.
function readable(c) {
  const lin = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const L = 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
  return L > 0.42 ? '#141414' : '#f2f2f2';
}

const el = id => document.getElementById(id);
const stage = el('stage'), shot = el('shot'), invite = el('invite');
const strip = el('strip'), stripEmpty = el('stripEmpty'), exports = el('exports');
const controls = el('controls'), kSlider = el('k'), kVal = el('kval');
const fileInput = el('file');
const pickerRow = el('pickerRow'), picker = el('picker');

let palette = [];
let rgbCache = null;    // sampled pixels as RGB
let labCache = null;    // the same pixels converted to Lab, built once
let space = 'lab';      // which one k-means runs in

/* User edits live alongside the computed clusters rather than
   inside them, so changing k or the colour space recomputes the
   algorithm's colours without discarding anything hand-picked. */
let overrides = new Map();   // cluster index -> [r, g, b]
let extras = [];             // colours added on top of the k clusters
let armed = null;            // {kind: 'cluster'|'extra', index} or null
let sampling = false;        // is the pointer currently an eyedropper?
let sampler = null;          // canvas holding the photo for eyedropping

// What the strip actually shows: clusters with any overrides applied,
// then the extras.
function displayList() {
  const fromClusters = palette.map((c, i) => {
    const o = overrides.get(i);
    return o
      ? { r: o[0], g: o[1], b: o[2], share: c.share, custom: true, kind: 'cluster', index: i }
      : { ...c, custom: false, kind: 'cluster', index: i };
  });
  const fromExtras = extras.map((e, i) => ({
    r: e[0], g: e[1], b: e[2], share: null, custom: true, kind: 'extra', index: i
  }));
  return fromClusters.concat(fromExtras);
}

function render() {
  strip.querySelectorAll('.band').forEach(n => n.remove());
  const list = displayList();
  if (!list.length) return;

  stripEmpty.style.display = 'none';
  exports.classList.remove('off');

  list.forEach(c => {
    const isArmed = armed && armed.kind === c.kind && armed.index === c.index;

    // A div rather than a button: bands contain their own buttons,
    // and nesting buttons is invalid HTML.
    const b = document.createElement('div');
    b.className = 'band' + (isArmed ? ' armed' : '');
    b.setAttribute('role', 'button');
    b.tabIndex = 0;
    b.style.background = hex(c);
    b.style.color = readable(c);
    b.style.flex = (0.4 + (c.share === null ? 0.12 : c.share * 4)).toFixed(3) + ' 1 0';
    b.title = 'Copy ' + hex(c);

    const h = document.createElement('span');
    h.className = 'band-hex';
    h.textContent = hex(c);

    const s = document.createElement('span');
    s.className = 'band-share';
    // A replaced colour no longer covers the share its cluster did,
    // so showing the old percentage would be a lie.
    s.textContent = c.custom ? 'custom' : (c.share * 100).toFixed(1) + '%';

    const tools = document.createElement('div');
    tools.className = 'band-tools';

    const edit = document.createElement('button');
    edit.className = 'tool' + (isArmed ? ' on' : '');
    edit.textContent = isArmed ? 'Done' : 'Change';
    edit.style.background = readable(c) === '#141414'
      ? 'rgba(255,255,255,0.72)' : 'rgba(20,20,20,0.34)';
    edit.setAttribute('aria-label', (isArmed ? 'Finish changing ' : 'Change ') + hex(c));
    edit.addEventListener('click', e => {
      e.stopPropagation();
      isArmed ? disarm() : arm(c.kind, c.index, hex(c));
    });
    tools.appendChild(edit);

    if (c.kind === 'extra' || overrides.has(c.index)) {
      const undo = document.createElement('button');
      undo.className = 'tool';
      undo.textContent = c.kind === 'extra' ? 'Remove' : 'Undo';
      undo.style.background = readable(c) === '#141414'
        ? 'rgba(255,255,255,0.72)' : 'rgba(20,20,20,0.34)';
      undo.addEventListener('click', e => {
        e.stopPropagation();
        if (c.kind === 'extra') extras.splice(c.index, 1);
        else overrides.delete(c.index);
        disarm();
      });
      tools.appendChild(undo);
    }

    const done = document.createElement('span');
    done.className = 'band-copied';
    done.style.background = hex(c);
    done.textContent = 'Copied';

    const doCopy = () => {
      copy(hex(c));
      done.classList.add('show');
      setTimeout(() => done.classList.remove('show'), 750);
    };

    b.append(h, s, tools, done);
    b.addEventListener('click', doCopy);
    b.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); doCopy(); }
    });
    strip.insertBefore(b, pickerRow);
  });
}

/* -----------------------------------------------------------------
   Editing a slot. Arming a band puts the app in sampling mode: the
   next click on the photo writes that pixel into the slot. The
   colour input is the manual alternative.
   ----------------------------------------------------------------- */
function arm(kind, index, currentHex) {
  armed = { kind, index };
  picker.value = currentHex.toLowerCase();
  pickerRow.classList.add('on');
  startSampling();
  render();
}

/* Sampling is a separate state from being armed. Arming opens the
   slot for editing; sampling is the brief moment the pointer is
   acting as an eyedropper. One click ends it and gives the normal
   cursor back, so Done and edit stay clickable. */
function startSampling() {
  if (!armed) return;
  sampling = true;
  stage.classList.add('sampling');
  el('pickerHint').textContent = 'Point at the photo and click.';
  el('resample').style.display = 'none';
  showHint('Click anywhere on the photo to take that colour');
}

function stopSampling() {
  sampling = false;
  stage.classList.remove('sampling');
  hideLoupe();
  el('resample').style.display = '';
  showHint(null);
}

function showHint(text) {
  const h = el('stageHint');
  if (!text) return h.classList.remove('on');
  h.textContent = text;
  h.classList.add('on');
}

function disarm() {
  armed = null;
  stopSampling();
  pickerRow.classList.remove('on');
  el('pickerHint').textContent = 'Point at the photo and click.';
  render();
}

function setArmed(rgb) {
  if (!armed) return;
  if (armed.kind === 'extra') extras[armed.index] = rgb;
  else overrides.set(armed.index, rgb);
  render();
}

function copy(text) {
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
  } else {
    fallbackCopy(text);
  }
}
function fallbackCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); } catch (e) {}
  ta.remove();
}

/* =================================================================
   4. WIRING
   ================================================================= */
function extract() {
  if (!rgbCache || !rgbCache.length) return;

  const working = space === 'lab' ? labCache : rgbCache;
  const clusters = kmeans(working, +kSlider.value);

  palette = clusters.map(({ c, share }) => {
    const [r, g, b] = space === 'lab'
      ? labToRgb(c[0], c[1], c[2])
      : [Math.round(c[0]), Math.round(c[1]), Math.round(c[2])];
    return { r, g, b, share };
  });

  render();
}

function load(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  shot.onload = () => {
    URL.revokeObjectURL(url);
    rgbCache = pixelsFrom(shot);
    // Converted once per photo, not once per extraction, so toggling
    // between spaces or dragging the slider stays instant.
    labCache = rgbCache.map(p => rgbToLab(p[0], p[1], p[2]));

    // A second, higher-resolution copy purely for eyedropping. The
    // 200px clustering sample is too coarse to hit a thin detail
    // like a strip of grass.
    sampler = document.createElement('canvas');
    const sc = Math.min(1, 1000 / Math.max(shot.naturalWidth, shot.naturalHeight));
    sampler.width = Math.max(1, Math.round(shot.naturalWidth * sc));
    sampler.height = Math.max(1, Math.round(shot.naturalHeight * sc));
    sampler.getContext('2d', { willReadFrequently: true })
           .drawImage(shot, 0, 0, sampler.width, sampler.height);

    overrides.clear();
    extras = [];
    disarm();
    shot.classList.add('on');
    invite.classList.add('off');
    controls.classList.add('on');
    extract();
  };
  shot.src = url;
}

el('pick').addEventListener('click', () => fileInput.click());
el('again').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => load(e.target.files[0]));

const rgbBtn = el('spaceRgb'), labBtn = el('spaceLab');

/* -----------------------------------------------------------------
   Eyedropper with a magnifier.

   The photo is displayed scaled, so a pointer position is converted
   to a fraction of the rendered box and back into sampler-canvas
   pixels. Both the loupe and the click use the same two helpers, so
   what the loupe shows is exactly what a click will take.
   ----------------------------------------------------------------- */
const N = 11, Z = 12, LOUPE_PX = N * Z;   // 11x11 pixels at 12x
const loupe = el('loupe'), loupeHex = el('loupeHex'), loupeChip = el('loupeChip');
const lctx = el('loupeCanvas').getContext('2d');

function pointToCanvas(e) {
  if (!sampler) return null;
  const r = shot.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width;
  const fy = (e.clientY - r.top) / r.height;
  if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null;
  return {
    cx: Math.min(sampler.width - 1, Math.floor(fx * sampler.width)),
    cy: Math.min(sampler.height - 1, Math.floor(fy * sampler.height))
  };
}

// A 3x3 average smooths over sensor noise and JPEG artefacts.
function sampleAt(cx, cy) {
  const x0 = Math.max(0, cx - 1), y0 = Math.max(0, cy - 1);
  const w = Math.min(3, sampler.width - x0), h = Math.min(3, sampler.height - y0);
  const d = sampler.getContext('2d').getImageData(x0, y0, w, h).data;
  let sr = 0, sg = 0, sb = 0, n = 0;
  for (let i = 0; i < d.length; i += 4) { sr += d[i]; sg += d[i + 1]; sb += d[i + 2]; n++; }
  return [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)];
}

function drawLoupe(cx, cy) {
  const half = (N - 1) / 2;
  // Clamp the source rect at the image edges and shift the
  // destination to match, so the centre cell stays under the cursor
  // instead of the view sliding inward.
  let sx = cx - half, sy = cy - half, sw = N, sh = N, dx = 0, dy = 0;
  if (sx < 0) { dx = -sx * Z; sw += sx; sx = 0; }
  if (sy < 0) { dy = -sy * Z; sh += sy; sy = 0; }
  if (sx + sw > sampler.width) sw = sampler.width - sx;
  if (sy + sh > sampler.height) sh = sampler.height - sy;

  lctx.imageSmoothingEnabled = false;
  lctx.fillStyle = '#2a2d33';
  lctx.fillRect(0, 0, LOUPE_PX, LOUPE_PX);
  if (sw > 0 && sh > 0) lctx.drawImage(sampler, sx, sy, sw, sh, dx, dy, sw * Z, sh * Z);

  // Two strokes so the marker survives on both light and dark pixels.
  const m = half * Z;
  lctx.lineWidth = 2;
  lctx.strokeStyle = 'rgba(0,0,0,0.85)';
  lctx.strokeRect(m - 1, m - 1, Z + 2, Z + 2);
  lctx.lineWidth = 1;
  lctx.strokeStyle = 'rgba(255,255,255,0.95)';
  lctx.strokeRect(m - 0.5, m - 0.5, Z + 1, Z + 1);
}

function moveLoupe(e) {
  const box = stage.getBoundingClientRect();
  let x = e.clientX - box.left + 22;
  let y = e.clientY - box.top + 22;
  // Flip to the other side rather than letting it hang off the stage.
  if (x + 136 > box.width) x = e.clientX - box.left - 158;
  if (y + 166 > box.height) y = e.clientY - box.top - 188;
  loupe.style.left = Math.max(4, x) + 'px';
  loupe.style.top = Math.max(4, y) + 'px';
}

function hideLoupe() { loupe.classList.remove('on'); }

shot.addEventListener('pointermove', e => {
  if (!armed || !sampling || !sampler) return hideLoupe();
  const p = pointToCanvas(e);
  if (!p) return hideLoupe();

  drawLoupe(p.cx, p.cy);
  const rgb = sampleAt(p.cx, p.cy);
  const hx = '#' + rgb.map(v => v.toString(16).padStart(2, '0')).join('').toUpperCase();
  loupeHex.textContent = hx;
  loupeChip.style.background = hx;
  moveLoupe(e);
  loupe.classList.add('on');
});

shot.addEventListener('pointerleave', hideLoupe);

shot.addEventListener('click', e => {
  if (!armed || !sampling || !sampler) return;
  e.stopPropagation();
  const p = pointToCanvas(e);
  if (!p) return;
  const rgb = sampleAt(p.cx, p.cy);
  const hx = '#' + rgb.map(v => v.toString(16).padStart(2, '0')).join('');
  picker.value = hx;
  setArmed(rgb);
  // Colour taken — hand the cursor back rather than staying in
  // eyedropper mode until the user dismisses it.
  stopSampling();
  el('pickerHint').textContent = 'Got ' + hx.toUpperCase() + '. Fine-tune it with the swatch, or pick again.';
});

el('resample').addEventListener('click', startSampling);

picker.addEventListener('input', () => {
  const v = picker.value;
  setArmed([
    parseInt(v.slice(1, 3), 16),
    parseInt(v.slice(3, 5), 16),
    parseInt(v.slice(5, 7), 16)
  ]);
});

el('doneEdit').addEventListener('click', disarm);

el('addColour').addEventListener('click', () => {
  // Seed a new slot from the largest cluster so the swatch is visible
  // straight away, then arm it for the user to change.
  const seedC = palette[0] || { r: 128, g: 128, b: 128 };
  extras.push([seedC.r, seedC.g, seedC.b]);
  render();
  arm('extra', extras.length - 1, hex({ r: seedC.r, g: seedC.g, b: seedC.b }));
});

window.addEventListener('keydown', e => { if (e.key === 'Escape' && armed) disarm(); });
function setSpace(next) {
  if (space === next) return;
  space = next;
  rgbBtn.classList.toggle('on', next === 'rgb');
  labBtn.classList.toggle('on', next === 'lab');
  extract();
}
rgbBtn.addEventListener('click', () => setSpace('rgb'));
labBtn.addEventListener('click', () => setSpace('lab'));

kSlider.addEventListener('input', () => { kVal.textContent = kSlider.value; });
kSlider.addEventListener('change', extract);

// drag and drop
['dragenter', 'dragover'].forEach(ev =>
  stage.addEventListener(ev, e => { e.preventDefault(); stage.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(ev =>
  stage.addEventListener(ev, e => { e.preventDefault(); stage.classList.remove('dragging'); }));
stage.addEventListener('drop', e => load(e.dataTransfer.files[0]));

// paste from clipboard
window.addEventListener('paste', e => {
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
  if (item) load(item.getAsFile());
});

el('copyCss').addEventListener('click', () => {
  copy(':root {\n' + displayList().map((c, i) => `  --colour-${i + 1}: ${hex(c)};`).join('\n') + '\n}');
});

/* Save the palette as a PNG. Same proportional bands as the strip,
   drawn at a size that survives being posted somewhere. */
el('saveImg').addEventListener('click', () => {
  const list = displayList();
  if (!list.length) return;

  const W = 1600, H = 900;
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');

  const weights = list.map(c => 0.4 + (c.share === null ? 0.12 : c.share * 4));
  const total = weights.reduce((a, b) => a + b, 0);

  ctx.font = '600 34px Archivo, "Helvetica Neue", Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';

  let x = 0;
  list.forEach((c, i) => {
    // Last band absorbs the rounding remainder so there is no seam.
    const w = i === list.length - 1 ? W - x : Math.round(W * weights[i] / total);
    ctx.fillStyle = hex(c);
    ctx.fillRect(x, 0, w, H);
    ctx.fillStyle = readable(c);
    ctx.fillText(hex(c), x + w / 2, H - 58);
    x += w;
  });

  cv.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'palette-' + hex(list[0]).slice(1).toLowerCase() + '.png';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }, 'image/png');
});