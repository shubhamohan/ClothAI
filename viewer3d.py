"""360-degree viewer that rotates the generated image itself (Three.js, runs in the browser)."""
import base64
import io

from PIL import Image

TEMPLATE = """
<div id="c" style="width:100%;height:540px;border-radius:16px;overflow:hidden;cursor:grab;touch-action:none;
     background:linear-gradient(180deg,#fdf2f8,#efeaff);"></div>
<div style="display:flex;gap:8px;justify-content:center;align-items:center;margin-top:8px;flex-wrap:wrap;
     font:13px Poppins,sans-serif;color:#6b5b7b">
  <label><input type="checkbox" id="auto" checked> auto-rotate</label>
</div>
<div style="text-align:center;font:12px Poppins,sans-serif;color:#9a8aa8;margin-top:4px">
Drag to rotate &middot; scroll to zoom</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const SRC = "__SRC__", ASPECT = __ASPECT__;
const el = document.getElementById('c');
if (typeof THREE === 'undefined') {
  el.innerHTML = '<img src="' + SRC + '" style="width:100%;border-radius:16px">' +
    '<p style="font-family:sans-serif;padding:8px">3D view could not load (needs internet). Showing the plain image.</p>';
} else {
  const W = el.clientWidth || 500, H = el.clientHeight;
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(35, W / H, 0.1, 100);
  cam.position.set(0, 0, 12);
  const ren = new THREE.WebGLRenderer({antialias: true, alpha: true});
  ren.setSize(W, H); ren.setPixelRatio(Math.min(2, window.devicePixelRatio));
  ren.outputEncoding = THREE.sRGBEncoding;
  el.appendChild(ren.domElement);

  const tex = new THREE.TextureLoader().load(SRC);
  tex.encoding = THREE.sRGBEncoding; tex.anisotropy = 8;
  const face = new THREE.MeshBasicMaterial({map: tex});
  const edge = new THREE.MeshBasicMaterial({color: 0xd9ccf2});
  const h = 6, w = h * ASPECT, d = 0.35;                       // slab: image on front and back, thin edge
  const slab = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), [edge, edge, edge, edge, face, face]);
  const grp = new THREE.Group(); grp.add(slab); scene.add(grp);
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(2.6, 48),
      new THREE.MeshBasicMaterial({color: 0x6d28d9, transparent: true, opacity: 0.12}));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = -h / 2 - 0.2; scene.add(shadow);

  let rot = 0, drag = false, lx = 0;
  const cv = ren.domElement, auto = document.getElementById('auto');
  cv.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; cv.setPointerCapture(e.pointerId); el.style.cursor = 'grabbing'; });
  cv.addEventListener('pointermove', e => { if (drag) { rot += (e.clientX - lx) * 0.012; lx = e.clientX; } });
  cv.addEventListener('pointerup', () => { drag = false; el.style.cursor = 'grab'; });
  cv.addEventListener('wheel', e => { e.preventDefault(); cam.position.z = Math.min(18, Math.max(7, cam.position.z + e.deltaY * 0.01)); }, {passive: false});
  (function loop() {
    requestAnimationFrame(loop);
    if (!drag && auto.checked) { rot += 0.008; }
    grp.rotation.y = rot;
    ren.render(scene, cam);
  })();
}
</script>
"""


def build_html(img):
    """img: PIL image. Returns HTML for st.components.v1.html."""
    im = img.convert("RGB")
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return TEMPLATE.replace("__SRC__", src).replace("__ASPECT__", f"{im.width / im.height:.4f}")