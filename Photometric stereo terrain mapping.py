import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # change to Qt5Agg or Agg if TkAgg not available
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa
from scipy.ndimage import gaussian_filter, uniform_filter
import sys, os, time, math

# ═══════════════════════════════════════════════════════════════
#  SETTINGS — edit these to match your setup
# ═══════════════════════════════════════════════════════════════
CAMERA_INDEX = 1       # webcam index (try 1 or 2 if 0 doesn't work)
CAP_W        = 640     # webcam capture width
CAP_H        = 480     # webcam capture height
PATCH_W      = 320     # width to store each patch (smaller = faster)
PATCH_H      = 240     # height to store each patch
NUM_LIGHTS   = 10      # number of light positions per rover stop
GRID_COLS    = 3       # rover stops per row in stitched terrain
COLORMAP     = "terrain"
SAVE_DIR     = "rover_ps_output"
# ═══════════════════════════════════════════════════════════════


# ── Light direction generation ─────────────────────────────────────────────
def generate_light_dirs(n, elevation_deg=45.0):
    """
    Generate N evenly-spaced light directions around a cone at
    'elevation_deg' above the horizontal plane (like a ring light).
    Returns list of n unit vectors (3,).
    """
    dirs = []
    el   = math.radians(elevation_deg)
    for i in range(n):
        az = 2 * math.pi * i / n          # azimuth angle
        lx = math.cos(el) * math.cos(az)
        ly = math.cos(el) * math.sin(az)
        lz = math.sin(el)
        v  = np.array([lx, ly, lz], dtype=np.float32)
        dirs.append(v / np.linalg.norm(v))
    return dirs

LIGHT_DIRS = generate_light_dirs(NUM_LIGHTS)


# ── Image utilities ────────────────────────────────────────────────────────
def to_gray_float(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32) / 255.0

def resize_to_patch(frame):
    return cv2.resize(frame, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)


# ── Photometric stereo core ────────────────────────────────────────────────
def compute_normals_ls(images, light_dirs):
    """
    Least-squares photometric stereo.
    images     : list of (H,W) float32  [0,1]
    light_dirs : list of (3,) float32 unit vectors
    Returns normals (H,W,3), albedo (H,W).
    """
    H, W = images[0].shape
    L    = np.array(light_dirs, dtype=np.float32)           # (N,3)
    I    = np.stack([im.flatten() for im in images])        # (N, H*W)

    Lp   = np.linalg.pinv(L)                                # (3,N)
    M    = Lp @ I                                           # (3, H*W)

    albedo = np.linalg.norm(M, axis=0).reshape(H, W)
    safe_a = np.where(albedo > 1e-6, albedo, 1e-6)

    normals = (M / safe_a.flatten()).T.reshape(H, W, 3).astype(np.float32)
    return normals, albedo.reshape(H, W).astype(np.float32)


def frankot_chellappa(normals):
    """
    Frequency-domain surface integration (Frankot & Chellappa 1988).
    normals : (H,W,3)
    Returns height map normalised to [0,1].
    """
    Nx, Ny, Nz = normals[...,0], normals[...,1], normals[...,2]
    eps   = 1e-6
    Nzs   = np.where(np.abs(Nz) > eps, Nz, eps)
    p, q  = -Nx / Nzs, -Ny / Nzs

    H, W  = p.shape
    fx    = np.fft.fftfreq(W)
    fy    = np.fft.fftfreq(H)
    FX, FY = np.meshgrid(fx, fy)

    P, Q  = np.fft.fft2(p), np.fft.fft2(q)
    denom = (2j*np.pi*FX)**2 + (2j*np.pi*FY)**2
    denom = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    Zf    = (2j*np.pi*FX*P + 2j*np.pi*FY*Q) / denom
    Zf[0,0] = 0

    Z = np.real(np.fft.ifft2(Zf)).astype(np.float32)
    Z = gaussian_filter(Z, sigma=2)
    Z -= Z.min()
    r  = Z.max() - Z.min()
    if r > 1e-6: Z /= r
    return Z


def single_image_height(gray):
    """
    Fallback: estimate height from a single image (texture + edges).
    Used when only 1 image is captured at a stop.
    """
    g   = gray.astype(np.float32) / 255.0 if gray.dtype == np.uint8 else gray
    mu  = uniform_filter(g, 9)
    mu2 = uniform_filter(g*g, 9)
    var = np.sqrt(np.clip(mu2 - mu**2, 0, None))
    low = gaussian_filter(g, sigma=20)
    gx  = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5)
    gy  = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    edg = gaussian_filter(np.sqrt(gx**2+gy**2), sigma=3)
    h   = 0.45*var + 0.35*low + 0.20*edg
    h   = gaussian_filter(h, sigma=2)
    h  -= h.min(); r = h.max()-h.min()
    if r > 1e-6: h /= r
    return h.astype(np.float32)


def calibrate_stop(images):
    """
    Given a list of captured images at one rover stop, compute height map.
    Uses photometric stereo if ≥3 images, otherwise texture-based fallback.
    """
    n = len(images)
    if n == 0:
        return np.zeros((PATCH_H, PATCH_W), dtype=np.float32), None

    grays = [to_gray_float(resize_to_patch(im)) for im in images]

    if n >= 3:
        dirs    = LIGHT_DIRS[:n]   # use first n directions
        normals, albedo = compute_normals_ls(grays, dirs)
        height  = frankot_chellappa(normals)
    else:
        # fallback: average the grayscales, use texture height
        avg     = np.mean(np.stack(grays), axis=0)
        height  = single_image_height(avg)
        normals = None

    return height, normals


# ── Terrain stitching ──────────────────────────────────────────────────────
def stitch_terrain(stop_heights):
    """
    Arrange per-stop height maps into a grid terrain.
    stop_heights : list of (PATCH_H, PATCH_W) float32
    """
    n    = len(stop_heights)
    cols = GRID_COLS
    rows = math.ceil(n / cols)
    blank = np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
    padded = stop_heights + [blank] * (rows * cols - n)

    rows_list = [np.hstack(padded[r*cols:(r+1)*cols]) for r in range(rows)]
    terrain   = np.vstack(rows_list)

    terrain = gaussian_filter(terrain, sigma=4)
    terrain -= terrain.min()
    r = terrain.max() - terrain.min()
    if r > 1e-6: terrain /= r
    return terrain


# ── Saving ─────────────────────────────────────────────────────────────────
def save_results(terrain, stop_heights, stop_normals):
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, "terrain.npy"), terrain)
    cv2.imwrite(os.path.join(SAVE_DIR, "terrain.png"),
                (terrain * 255).astype(np.uint8))
    for i, h in enumerate(stop_heights):
        cv2.imwrite(os.path.join(SAVE_DIR, f"stop_{i+1:02d}_height.png"),
                    (h * 255).astype(np.uint8))
        if stop_normals[i] is not None:
            nm = ((stop_normals[i]+1)/2*255).clip(0,255).astype(np.uint8)
            cv2.imwrite(os.path.join(SAVE_DIR, f"stop_{i+1:02d}_normals.png"),
                        cv2.cvtColor(nm, cv2.COLOR_RGB2BGR))
    print(f"[SAVED]  '{SAVE_DIR}/'")


# ── 3-D render ─────────────────────────────────────────────────────────────
def render_terrain(terrain, stop_heights, stop_normals):
    TH, TW = terrain.shape
    stride = max(1, min(TH, TW) // 100)
    Zs = terrain[::stride, ::stride]
    X, Y = np.meshgrid(np.linspace(0, TW, Zs.shape[1]),
                       np.linspace(0, TH, Zs.shape[0]))

    fig = plt.figure(figsize=(18, 10), facecolor='#111118')
    fig.suptitle(
        f"Rover Photometric Stereo Terrain  —  "
        f"{len(stop_heights)} stop(s), {NUM_LIGHTS} lights/stop",
        color='white', fontsize=14, y=0.99
    )

    # 3-D surface
    ax3 = fig.add_subplot(2, 3, (1, 4), projection='3d')
    ax3.set_facecolor('#09090f')
    surf = ax3.plot_surface(X, Y, Zs, cmap=plt.get_cmap(COLORMAP),
                             linewidth=0, antialiased=True, alpha=0.93)
    ax3.set_xlabel('X', color='#777'); ax3.set_ylabel('Y', color='#777')
    ax3.set_zlabel('Height', color='#777')
    ax3.set_title('3D Terrain Surface', color='white', pad=10)
    ax3.tick_params(colors='#555')
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False
    cb = fig.colorbar(surf, ax=ax3, shrink=0.45, aspect=14, pad=0.07)
    cb.set_label('Relative Height', color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    ax3.view_init(elev=38, azim=-55)

    def _panel(pos, img, title, cmap='gray', cbar=False):
        ax = fig.add_subplot(*pos)
        ax.set_facecolor('#09090f')
        im = ax.imshow(img, cmap=cmap, aspect='auto')
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
        if cbar:
            cb2 = fig.colorbar(im, ax=ax, shrink=0.8)
            plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')

    _panel((2,3,5), terrain, 'Full Height Map', cbar=True)

    # Topographic contour
    ax_ct = fig.add_subplot(2, 3, 6)
    ax_ct.set_facecolor('#09090f')
    ctf = ax_ct.contourf(terrain, levels=25, cmap='terrain')
    ax_ct.contour(terrain, levels=25, colors='white', linewidths=0.3, alpha=0.3)
    ax_ct.set_title('Topographic Contour', color='white', fontsize=10)
    ax_ct.axis('off')
    cb3 = fig.colorbar(ctf, ax=ax_ct, shrink=0.8)
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout()

    # Per-stop normals (if available)
    valid_normals = [(i,n) for i,n in enumerate(stop_normals) if n is not None]
    if valid_normals:
        nc  = min(len(valid_normals), 4)
        fig2, axes = plt.subplots(1, nc, figsize=(4*nc, 4), facecolor='#111118')
        fig2.suptitle('Surface Normal Maps per Stop', color='white', fontsize=12)
        if nc == 1: axes = [axes]
        for ax, (i, nm) in zip(axes, valid_normals[:nc]):
            nm_rgb = ((nm+1)/2).clip(0,1)
            ax.imshow(nm_rgb)
            ax.set_title(f'Stop {i+1}', color='white', fontsize=9)
            ax.axis('off')
        plt.tight_layout()

    print("\n[INFO] 3D terrain window open.")
    print("       Rotate  : left-click drag")
    print("       Zoom    : scroll wheel")
    print("       Close to exit.\n")
    plt.show()


# ── HUD overlay ────────────────────────────────────────────────────────────
def draw_hud(frame, state):
    """
    state dict keys:
      camera_on, stop_num, captured, num_lights, calibrated_stops,
      flash, show_ring
    """
    h, w    = frame.shape[:2]
    cam     = state['camera_on']
    stop    = state['stop_num']          # 1-based
    cap     = state['captured']          # captures this stop
    nl      = state['num_lights']
    cal     = state['calibrated_stops']  # number of stops done

    # ── Top bar ──────────────────────────────────────────────────────────
    bar = frame.copy()
    cv2.rectangle(bar, (0,0), (w,68), (8,8,14), -1)
    cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

    if not cam:
        cv2.putText(frame, "Press  SPACEBAR  to start camera",
                    (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (60,210,255), 2)
    else:
        # Live dot
        dot_col = (0,80,255) if int(time.time()*2)%2 else (0,40,130)
        cv2.circle(frame, (w-22, 22), 9, dot_col, -1)
        cv2.putText(frame, "REC", (w-60, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,80,80), 1)

        cv2.putText(frame,
                    f"STOP {stop}  |  Light {cap}/{nl}  |  Stops calibrated: {cal}",
                    (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80,240,80), 2)
        cv2.putText(frame,
                    "SPACE=capture   C=calibrate stop   T=build terrain   R=reset stop   Q=quit",
                    (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,180,180), 1)

    # ── Light ring visualisation ──────────────────────────────────────────
    if cam and state.get('show_ring', True):
        cx, cy, radius = 58, h-58, 44
        cv2.circle(frame, (cx,cy), radius, (60,60,80), 1)
        for i in range(nl):
            angle = 2*math.pi*i/nl - math.pi/2
            lx = int(cx + radius * math.cos(angle))
            ly = int(cy + radius * math.sin(angle))
            if i < cap:
                cv2.circle(frame, (lx,ly), 6, (60,230,60), -1)   # captured
            elif i == cap:
                cv2.circle(frame, (lx,ly), 7, (255,200,0), -1)   # next
            else:
                cv2.circle(frame, (lx,ly), 4, (80,80,100), -1)   # pending
        cv2.putText(frame, "ring", (cx-14, cy+58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130,130,160), 1)

    # ── Bottom status bar ─────────────────────────────────────────────────
    if cam and cap == nl:
        cv2.rectangle(frame, (0, h-30), (w, h), (20,60,20), -1)
        cv2.putText(frame,
                    f"  All {nl} lights captured!  Press C to calibrate this stop.",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,255,100), 1)
    elif cam and cal > 0:
        cv2.rectangle(frame, (0, h-30), (w, h), (20,20,50), -1)
        cv2.putText(frame,
                    f"  {cal} stop(s) calibrated.  Press T to build full terrain.",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140,180,255), 1)

    return frame


def flash_frame(frame):
    white = np.ones_like(frame) * 255
    return cv2.addWeighted(frame, 0.25, white, 0.75, 0)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Rover Photometric Stereo — Rotating Light Terrain Mapper")
    print("=" * 62)
    print(f"  Lights per stop : {NUM_LIGHTS}")
    print(f"  Grid columns    : {GRID_COLS}")
    print()
    print("  SPACEBAR  start camera / capture (one per light position)")
    print("  C         calibrate current rover stop")
    print("  T         build full terrain from all calibrated stops")
    print("  R         reset current stop")
    print("  Q/ESC     quit\n")

    vcap            = None
    camera_on       = False
    flash_count     = 0

    # Per-stop state
    stop_images     = []    # images captured at current stop
    stop_num        = 1     # current stop index (1-based)

    # Accumulated results
    all_heights     = []    # one height map per calibrated stop
    all_normals     = []    # one normal map (or None) per stop

    cv2.namedWindow("Rover PS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rover PS", CAP_W, CAP_H)
    blank = np.zeros((CAP_H, CAP_W, 3), dtype=np.uint8)

    while True:
        # Read frame
        if camera_on and vcap is not None:
            ret, frame = vcap.read()
            if not ret:
                print("[ERROR] Camera read failed."); break
        else:
            frame = blank.copy()

        if flash_count > 0:
            frame = flash_frame(frame)
            flash_count -= 1

        state = dict(
            camera_on        = camera_on,
            stop_num         = stop_num,
            captured         = len(stop_images),
            num_lights       = NUM_LIGHTS,
            calibrated_stops = len(all_heights),
            show_ring        = True,
        )
        display = draw_hud(frame.copy(), state)
        cv2.imshow("Rover PS", display)
        key = cv2.waitKey(30) & 0xFF

        # ── Quit ──────────────────────────────────────────────────────────
        if key in (ord('q'), 27):
            print("[INFO] Quit."); break

        # ── SPACEBAR ──────────────────────────────────────────────────────
        if key == ord(' '):
            if not camera_on:
                vcap = cv2.VideoCapture(CAMERA_INDEX)
                if not vcap.isOpened():
                    sys.exit(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
                vcap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
                vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
                camera_on = True
                print("[READY] Camera on.")
                print(f"[STOP {stop_num}] Rotate light to position 1, press SPACEBAR.")
            else:
                n_cap = len(stop_images)
                if n_cap >= NUM_LIGHTS:
                    print(f"[WARN] Already captured {NUM_LIGHTS} images for stop {stop_num}."
                          " Press C to calibrate or R to reset.")
                else:
                    # Capture frame
                    ret2, raw = vcap.read()
                    if ret2:
                        stop_images.append(raw.copy())
                        flash_count = 5
                        n_cap += 1
                        print(f"  [CAPTURE] Stop {stop_num} — light position {n_cap}/{NUM_LIGHTS}")
                        if n_cap < NUM_LIGHTS:
                            print(f"            Rotate to position {n_cap+1}, press SPACEBAR.")
                        else:
                            print(f"  [DONE]    All {NUM_LIGHTS} lights captured."
                                  " Press C to calibrate.")

        # ── C — calibrate this stop ────────────────────────────────────────
        if key == ord('c') or key == ord('C'):
            if len(stop_images) == 0:
                print("[WARN] No images captured at this stop yet.")
            else:
                n_cap = len(stop_images)
                print(f"\n[CALIBRATE] Stop {stop_num}: computing height from"
                      f" {n_cap} image(s)...")
                height, normals = calibrate_stop(stop_images)
                all_heights.append(height)
                all_normals.append(normals)
                method = "photometric stereo" if n_cap >= 3 else "texture fallback"
                print(f"[CALIBRATE] Done ({method}).")
                print(f"[CALIBRATE] Stop {stop_num} saved. "
                      f"Total stops: {len(all_heights)}\n")

                # Show normal map preview briefly
                if normals is not None:
                    nm_bgr = cv2.cvtColor(
                        ((normals+1)/2*255).clip(0,255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)
                    nm_bgr = cv2.resize(nm_bgr, (CAP_W, CAP_H))
                    cv2.putText(nm_bgr,
                                f"Normal Map — Stop {stop_num}  (any key to continue)",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255,255,255), 1)
                    cv2.imshow("Rover PS", nm_bgr)
                    cv2.waitKey(0)

                # Reset for next rover stop
                stop_images = []
                stop_num   += 1
                print(f"[NEXT] Move rover to stop {stop_num}."
                      " Rotate light to position 1, press SPACEBAR.\n")

        # ── R — reset current stop ─────────────────────────────────────────
        if key == ord('r') or key == ord('R'):
            if stop_images:
                print(f"[RESET] Discarded {len(stop_images)} image(s) for"
                      f" stop {stop_num}. Start again.")
                stop_images = []
            else:
                print("[RESET] Nothing to reset at current stop.")

        # ── T — build full terrain ─────────────────────────────────────────
        if key == ord('t') or key == ord('T'):
            if len(all_heights) == 0:
                print("[WARN] No stops calibrated yet. "
                      "Calibrate at least one stop first.")
            else:
                print(f"\n[TERRAIN] Stitching {len(all_heights)} stop(s)...")
                if vcap: vcap.release()
                cv2.destroyAllWindows()

                terrain = stitch_terrain(all_heights)
                save_results(terrain, all_heights, all_normals)
                render_terrain(terrain, all_heights, all_normals)
                return

    if vcap: vcap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()