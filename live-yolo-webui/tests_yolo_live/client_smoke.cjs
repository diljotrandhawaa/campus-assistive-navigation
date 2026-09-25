/* Node 22+ test harness for the actual app.js with a simulated camera/DOM.
   Runs against mock_server.py; does not test a physical camera or browser painting. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const modeFile = '/tmp/yolo-ui-test-mode';
const image = fs.readFileSync(process.argv[2]);
const timers = new Set();
const every = (fn, ms) => { const timer = setInterval(fn, ms); timers.add(timer); return timer; };
const after = (fn, ms) => { const timer = setTimeout(fn, ms); timers.add(timer); return timer; };

class Element extends EventTarget {
  constructor() { super(); this.value = ''; this.textContent = ''; this.hidden = false;
    this.style = {}; this.dataset = {}; this.disabled = false; this.children = [];
    this.classList = { add(){}, remove(){}, toggle(){} };
  }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  click() { this.dispatchEvent(new Event('click')); }
  getContext() { return { drawImage(){}, fillRect(){}, fillText(){}, strokeRect(){},
    beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, measureText(t){ return {width:t.length*7}; } }; }
  toBlob(fn) { after(() => fn(new Blob([image], {type: 'image/jpeg'})), 2); }
}
const ids = new Map();
const get = id => { if (!ids.has(id)) ids.set(id, new Element()); return ids.get(id); };
get('confidence').value = '.25'; get('target').value = 'whiteboard'; get('rate').value = '8'; get('size').value = '960';
const video = get('video'); video.readyState = 4; video.videoWidth = 320; video.videoHeight = 240;
video.play = async () => {}; video.requestVideoFrameCallback = cb => after(cb, 30);
video.cancelVideoFrameCallback = clearTimeout;
const document = new EventTarget();
Object.assign(document, {hidden: false, getElementById: get, createElement: () => new Element()});
const media = new EventTarget();
media.enumerateDevices = async () => [{kind: 'videoinput', deviceId: 'test-camera', label: 'Test camera'}];
media.getUserMedia = async () => {
  const track = new EventTarget(); track.stop = () => {}; track.getSettings = () => ({deviceId:'test-camera'});
  return {getTracks: () => [track], getVideoTracks: () => [track]};
};
const sandbox = {console, document, navigator: {mediaDevices: media}, window: new EventTarget(),
  location: {protocol: 'http:', host: '127.0.0.1:18091'}, performance, Blob, WebSocket, URL,
  setTimeout: after, clearTimeout, setInterval: every, clearInterval,
  Option: function(label,value) {this.label=label;this.value=value;} };
const context = vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../yolo_live/static/app.js'), 'utf8'), context);
async function waitFor(check, label, timeout = 4500) {
  const until = performance.now() + timeout;
  while (!check()) {
    if (performance.now() > until) throw new Error(`Timed out: ${label}; UI=${get('instruction').textContent}; error=${get('error').textContent}`);
    await sleep(25);
  }
}
const setMode = mode => fs.writeFileSync(modeFile, mode);
const state = () => get('direction-card').dataset.state;
async function checkState(mode, expected = mode) {
  setMode(mode);
  await waitFor(() => state() === expected, expected);
  console.log('PASS', expected, get('instruction').textContent);
}

(async () => {
  setMode('right');
  get('start').click();
  await waitFor(() => state() === 'right', 'start -> right');
  assert.match(get('latency').textContent, /ms/);
  assert.equal(get('result').hidden, false);
  console.log('PASS start, camera frame upload, boxes, metrics');
  await checkState('centered');
  await checkState('left');
  await checkState('lost');
  assert.equal(get('marker').hidden, true);
  setMode('right'); get('target').value = 'door'; get('target').dispatchEvent(new Event('change'));
  await waitFor(() => state() === 'lost' && get('instruction').textContent.includes('Door'), 'target change');
  console.log('PASS target change does not reuse previous guidance');
  get('target').value = 'whiteboard'; get('target').dispatchEvent(new Event('change'));
  await waitFor(() => state() === 'right', 'target restored');
  setMode('slow');
  await waitFor(() => state() === 'stale', 'expired result');
  assert.equal(get('save').disabled, true);
  console.log('PASS stale results clear direction and disable snapshot');
  await sleep(2200);
  assert.equal(state(), 'stale');
  setMode('centered');
  await waitFor(() => state() === 'centered', 'recover after slow inference');
  document.hidden = true; document.dispatchEvent(new Event('visibilitychange'));
  assert.equal(state(), 'stale');
  await sleep(200);
  assert.equal(state(), 'stale');
  document.hidden = false; document.dispatchEvent(new Event('visibilitychange'));
  await waitFor(() => state() === 'centered', 'tab resume');
  console.log('PASS hidden tab clears guidance and resumes with a fresh frame');
  get('stop').click();
  assert.equal(state(), 'idle');
  assert.equal(get('preview-wrap').hidden, true);
  setMode('right'); get('start').click();
  await waitFor(() => state() === 'right', 'restart');
  console.log('PASS stop and restart');
  setMode('error');
  await waitFor(() => get('status').textContent === 'Analysis stopped', 'inference error');
  assert.match(get('error').textContent, /inference failed/);
  console.log('PASS server error stops camera and surfaces error');
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => {
  vm.runInContext('stop()', context);
  for (const timer of timers) { clearTimeout(timer); clearInterval(timer); }
  fs.rmSync(modeFile, {force:true});
});
