const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const staticDir = path.join(__dirname, '..', 'yolo_live', 'static');
const html = fs.readFileSync(path.join(staticDir, 'index.html'), 'utf8');
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(match => match[1]);
const intervals = [];
let clock = 100;
function element() {
  return {textContent: '', hidden: false, value: '', checked: true, disabled: false,
    children: [], dataset: {}, style: {},
    classList: {add(){}, remove(){}, toggle(){}},
    addEventListener(){}, append(...nodes){this.children.push(...nodes);},
    replaceChildren(...nodes){this.children = nodes;},
    getContext(){return new Proxy({measureText: t => ({width: t.length * 8})},
      {get(target, key){return target[key] || (() => {});}});},
  };
}
const elements = Object.fromEntries(ids.map(id => [id, element()]));
elements.target.value = 'door'; elements.confidence.value = '0.25';
elements.rate.value = '8'; elements.size.value = '1280';
const sandbox = {console, performance: {now: () => clock},
  document: {hidden: false, getElementById(id){assert.ok(elements[id], `Missing DOM id ${id}`); return elements[id];},
    createElement: element, addEventListener(){}},
  navigator: {}, window: {addEventListener(){}},
  setInterval: fn => intervals.push(fn), setTimeout: () => 1, clearTimeout(){},
  Option: function(text, value){this.text = text; this.value = value;},
  WebSocket: {OPEN: 1},
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(staticDir, 'app.js'), 'utf8'), sandbox);
vm.runInContext(`
  const testImage = {width: 100, height: 80};
  const testFrame = {id: 1, revision: 1, captured: 0, image: testImage};
  const testOcr = {status: 'ok', items: [{text: '<script>204</script>', confidence: .93,
    polygon: [[.1,.1],[.5,.1],[.5,.3],[.1,.3]], box:[.1,.1,.5,.3], position:'top-left',
    nearby_object:{label:'door'}}], ms: 20};
  renderOcr(testOcr, testFrame);
`, sandbox);
assert.equal(elements['ocr-frame'].hidden, false);
assert.equal(elements['ocr-readings'].children[0].children[0].textContent, '<script>204</script>');
assert.match(elements['ocr-state'].textContent, /frame 1/);
vm.runInContext("renderOcr({status:'skipped'}, {id:2});", sandbox);
assert.match(elements['ocr-state'].textContent, /frame 1/); // No attaching old boxes to frame 2.
clock = 5500;
vm.runInContext('running = true;', sandbox);
intervals[0]();
assert.equal(elements['ocr-frame'].hidden, true);
assert.equal(elements['ocr-readings'].children.length, 0);
clock = 2000;
vm.runInContext(`
  revision = 1; pending = testFrame; lastCameraFrame = 2000;
  handleMessage({type:'result', id:1, revision:1, detector_ms:2, ocr:testOcr});
`, sandbox);
assert.match(elements.instruction.textContent, /too old/); // 2s OCR snapshot accepted, guidance withheld.
assert.equal(elements['ocr-frame'].hidden, false);
vm.runInContext("renderOcr({status:'error',message:'OCR failed'},testFrame);", sandbox);
assert.equal(elements['ocr-frame'].hidden, true);
assert.equal(elements['ocr-state'].textContent, 'OCR failed');
vm.runInContext("handleMessage({type:'ready',classes:['door','elevator'],ocr_enabled:true,ocr_interval:1});", sandbox);
assert.equal(elements.target.children.length, 2);
console.log('UI checks passed: frame pairing, text escaping, expiry, stale guidance, error clearing, class synchronization.');
