/**
 * SORT | ARM AI - Dashboard Client v2.1
 * Premium HUD dashboard with real-time stats, bin counters, toast notifications,
 * session timer, live log streaming, and full pipeline control.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── State ──────────────────────────────────────────────────────────────────
  let isPreviewRunning = false;
  let isPipelineRunning = false;
  let activeModel = '';
  let lastLogTimestamp = '';
  let editingModel = '';
  let sessionStartTime = Date.now();
  let totalSorted = 0;
  let binCounts = { 1: 0, 2: 0, 3: 0 };
  let logEntryCount = 0;

  // ── DOM Refs ───────────────────────────────────────────────────────────────
  const previewImg            = document.getElementById('previewImg');
  const placeholderViewfinder = document.getElementById('placeholderViewfinder');
  const feedStatusBadge       = document.getElementById('feedStatusBadge');
  const btnStartPreview       = document.getElementById('btnStartPreview');
  const btnStopPreview        = document.getElementById('btnStopPreview');
  const confSlider            = document.getElementById('confSlider');
  const confValueDisplay      = document.getElementById('confValueDisplay');
  const hudModelName          = document.getElementById('hudModelName');

  const dropzone              = document.getElementById('dropzone');
  const modelFileInput        = document.getElementById('modelFileInput');
  const modelListContainer    = document.getElementById('modelListContainer');
  const modelCountBadge       = document.getElementById('modelCountBadge');
  const dropzoneTitle         = document.getElementById('dropzoneTitle');

  const activeModelDisplay    = document.getElementById('activeModelDisplay');
  const mockArmSwitch         = document.getElementById('mockArmSwitch');
  const pipelineStatusBanner  = document.getElementById('pipelineStatusBanner');
  const btnStartSorting       = document.getElementById('btnStartSorting');
  const btnStopSorting        = document.getElementById('btnStopSorting');

  const terminalBox           = document.getElementById('terminalBox');
  const btnClearLogs          = document.getElementById('btnClearLogs');
  const terminalLogCount      = document.getElementById('terminalLogCount');

  const mappingModal          = document.getElementById('mappingModal');
  const modalModelName        = document.getElementById('modalModelName');
  const mappingList           = document.getElementById('mappingList');
  const btnCloseModal         = document.getElementById('btnCloseModal');
  const btnCancelModal        = document.getElementById('btnCancelModal');
  const btnSaveMapping        = document.getElementById('btnSaveMapping');
  const btnSaveCoords         = document.getElementById('btnSaveCoords');

  const statTotalSorted       = document.getElementById('statTotalSorted');
  const statSessionTime       = document.getElementById('statSessionTime');

  const meterBin1             = document.getElementById('meterBin1');
  const meterBin2             = document.getElementById('meterBin2');
  const meterBin3             = document.getElementById('meterBin3');
  const countBin1             = document.getElementById('countBin1');
  const countBin2             = document.getElementById('countBin2');
  const countBin3             = document.getElementById('countBin3');

  const toastContainer        = document.getElementById('toastContainer');

  // ── Toast Notification ────────────────────────────────────────────────────
  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { success: '✓', error: '✕', info: 'ℹ' };
    toast.innerHTML = `<span style="font-size:14px;font-weight:800">${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.transition = 'opacity 0.4s, transform 0.4s';
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      setTimeout(() => toast.remove(), 400);
    }, duration);
  }

  // ── Session Timer ─────────────────────────────────────────────────────────
  function updateSessionTimer() {
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    if (statSessionTime) statSessionTime.textContent = `${m}:${s}`;
  }
  setInterval(updateSessionTimer, 1000);

  // ── Bin Counters ──────────────────────────────────────────────────────────
  function updateBinMeters() {
    const max = Math.max(1, Math.max(...Object.values(binCounts)));
    const update = (id, meter, count) => {
      const c = binCounts[id] || 0;
      if (meter) meter.style.width = `${Math.round((c / max) * 100)}%`;
      if (count) count.textContent = c;
    };
    update(1, meterBin1, countBin1);
    update(2, meterBin2, countBin2);
    update(3, meterBin3, countBin3);
    if (statTotalSorted) statTotalSorted.textContent = totalSorted;
  }

  // ── 1. VIDEO PREVIEW & CAMERA CONTROLS ─────────────────────────────────────
  const cameraSelect       = document.getElementById('cameraSelect');
  const btnRefreshCameras  = document.getElementById('btnRefreshCameras');
  const btnSwitchToCamera  = document.getElementById('btnSwitchToCamera');
  const btnModeAuto        = document.getElementById('btnModeAuto');
  const btnModeManual      = document.getElementById('btnModeManual');

  function updatePreviewUI(running) {
    isPreviewRunning = running;
    if (running) {
      previewImg.src = '/video_feed?' + Date.now();
      previewImg.style.display = 'block';
      placeholderViewfinder.style.display = 'none';
      feedStatusBadge.textContent = '● LIVE';
      feedStatusBadge.className = 'badge badge-live';
      btnStartPreview.disabled = true;
      btnStopPreview.disabled = false;
      if (hudModelName) hudModelName.textContent = `MODEL: ${activeModel || '---'}`;
    } else {
      previewImg.src = '';
      previewImg.style.display = 'none';
      placeholderViewfinder.style.display = 'flex';
      feedStatusBadge.textContent = '● OFFLINE';
      feedStatusBadge.className = 'badge badge-off';
      btnStartPreview.disabled = false;
      btnStopPreview.disabled = true;
    }
  }

  async function loadCameras() {
    try {
      const res = await fetch('/api/cameras');
      const data = await res.json();
      if (cameraSelect && data.cameras) {
        cameraSelect.innerHTML = '';
        data.cameras.forEach(cam => {
          const opt = document.createElement('option');
          opt.value = cam.index;
          opt.textContent = cam.name;
          if (cam.index === data.current_index) opt.selected = true;
          cameraSelect.appendChild(opt);
        });
        if (data.cameras.length === 0) {
          cameraSelect.innerHTML = '<option value="0">Camera 0 (Default)</option>';
        }
      }
    } catch (e) { console.warn('Camera scan failed'); }
  }

  if (cameraSelect) {
    cameraSelect.addEventListener('change', async (e) => {
      const idx = parseInt(e.target.value);
      try {
        await fetch('/api/preview/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ camera_index: idx })
        });
        if (btnSwitchToCamera) btnSwitchToCamera.style.display = 'none';
        if (testImageName) testImageName.textContent = '';
        updatePreviewUI(true);
        showToast(`Switched to Camera ${idx}`, 'info');
      } catch (err) { showToast('Camera switch failed', 'error'); }
    });
  }

  if (btnRefreshCameras) {
    btnRefreshCameras.addEventListener('click', async () => {
      btnRefreshCameras.innerHTML = '&#8987;';
      await loadCameras();
      btnRefreshCameras.innerHTML = '&#8635;';
      showToast('Camera list refreshed', 'info');
    });
  }

  if (btnSwitchToCamera) {
    btnSwitchToCamera.addEventListener('click', async () => {
      try {
        await fetch('/api/preview/switch_to_camera', { method: 'POST' });
        if (testImageName) testImageName.textContent = '';
        btnSwitchToCamera.style.display = 'none';
        updatePreviewUI(true);
        showToast('Switched back to Live Camera stream', 'success');
      } catch (err) { showToast('Failed to switch to camera', 'error'); }
    });
  }

  // ── Sorting Mode Toggle (Automatic vs Manual) ──────────────────────────────
  async function setSortingMode(mode) {
    try {
      const res = await fetch('/api/arm/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      const data = await res.json();
      if (data.success) {
        if (mode === 'auto') {
          if (btnModeAuto) btnModeAuto.classList.add('active');
          if (btnModeManual) btnModeManual.classList.remove('active');
          showToast('Sorting Mode: ⚡ AUTOMATIC (Auto Pick & Place on detection)', 'success');
        } else {
          if (btnModeManual) btnModeManual.classList.add('active');
          if (btnModeAuto) btnModeAuto.classList.remove('active');
          showToast('Sorting Mode: 🖱 MANUAL (Interactive drag or Demo Sort)', 'info');
        }
      }
    } catch (e) { console.warn('Mode toggle failed'); }
  }

  if (btnModeAuto && btnModeManual) {
    btnModeAuto.addEventListener('click', () => setSortingMode('auto'));
    btnModeManual.addEventListener('click', () => setSortingMode('manual'));
  }

  btnStartPreview.addEventListener('click', async () => {
    try {
      const idx = parseInt(cameraSelect?.value || 0);
      const res = await fetch('/api/preview/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: idx })
      });
      const data = await res.json();
      if (data.success) {
        if (btnSwitchToCamera) btnSwitchToCamera.style.display = 'none';
        if (testImageName) testImageName.textContent = '';
        updatePreviewUI(true);
        showToast(`Camera ${data.camera_index ?? 0} preview started`, 'success');
      }
    } catch (e) { showToast('Failed to start preview', 'error'); }
  });

  btnStopPreview.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/preview/stop', { method: 'POST' });
      const data = await res.json();
      if (data.success) { updatePreviewUI(false); showToast('Camera preview stopped', 'info'); }
    } catch (e) { showToast('Failed to stop preview', 'error'); }
  });

  confSlider.addEventListener('input', (e) => {
    confValueDisplay.textContent = `${e.target.value}%`;
  });

  confSlider.addEventListener('change', async (e) => {
    try {
      await fetch('/api/confidence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confidence: parseFloat(e.target.value) / 100.0 })
      });
    } catch (e) { console.warn('Confidence update failed'); }
  });

  // ── Test Image Upload ──────────────────────────────────────────────────────
  const testImageInput  = document.getElementById('testImageInput');
  const btnUploadTest   = document.getElementById('btnUploadTestImage');
  const btnLoadSample   = document.getElementById('btnLoadSampleImage');
  const testImageName   = document.getElementById('testImageName');

  if (btnUploadTest) {
    btnUploadTest.addEventListener('click', () => testImageInput.click());
    testImageInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('image', file);
      try {
        btnUploadTest.textContent = 'Loading...';
        const res = await fetch('/api/preview/upload_image', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.success) {
          if (testImageName) testImageName.textContent = `✓ ${file.name}`;
          if (btnSwitchToCamera) btnSwitchToCamera.style.display = 'inline-block';
          updatePreviewUI(true);
          showToast(`"${file.name}" loaded — model running on it!`, 'success');
        } else {
          showToast('Image upload failed: ' + (data.error || 'Unknown'), 'error');
        }
      } catch (err) {
        showToast('Failed to upload image', 'error');
      } finally {
        btnUploadTest.innerHTML = '&#128247; Upload Test Image';
        testImageInput.value = '';
      }
    });
  }

  if (btnLoadSample) {
    btnLoadSample.addEventListener('click', async () => {
      try {
        btnLoadSample.textContent = 'Loading...';
        const res = await fetch('/api/preview/load_sample', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          if (testImageName) testImageName.textContent = "✓ Sample Rubik's Cube active";
          if (btnSwitchToCamera) btnSwitchToCamera.style.display = 'inline-block';
          updatePreviewUI(true);
          showToast("Sample Rubik's Cube loaded — model running live inference!", 'success');
        } else {
          showToast('Failed to load sample: ' + (data.error || 'Unknown'), 'error');
        }
      } catch (err) {
        showToast('Failed to load sample image', 'error');
      } finally {
        btnLoadSample.innerHTML = '&#129513; Use Sample Cube Image';
      }
    });
  }

  // ── 2. MODEL UPLOAD ───────────────────────────────────────────────────────
  dropzone.addEventListener('click', () => modelFileInput.click());

  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) uploadModelFile(e.dataTransfer.files[0]);
  });

  modelFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) uploadModelFile(e.target.files[0]);
  });

  async function uploadModelFile(file) {
    const ext = file.name.toLowerCase();
    if (!ext.endsWith('.pt') && !ext.endsWith('.engine')) {
      showToast('Only .pt or .engine files are supported', 'error');
      return;
    }
    const formData = new FormData();
    formData.append('model', file);
    try {
      dropzoneTitle.textContent = `Uploading ${file.name}...`;
      const res = await fetch('/api/models/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        showToast(`${file.name} uploaded & set active`, 'success');
        loadModels();
      } else {
        showToast('Upload failed: ' + (data.error || 'Unknown'), 'error');
      }
    } catch (e) {
      showToast('Upload failed — check console', 'error');
    } finally {
      dropzoneTitle.textContent = 'Drop .pt / .engine file here';
    }
  }

  // ── 3. MODELS LIST ────────────────────────────────────────────────────────
  async function loadModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();

      activeModel = data.active_model || '';
      activeModelDisplay.textContent = activeModel || 'None Selected';
      if (hudModelName) hudModelName.textContent = `MODEL: ${activeModel || '---'}`;

      const models = data.models || [];
      if (modelCountBadge) modelCountBadge.textContent = `${models.length} model${models.length !== 1 ? 's' : ''}`;

      modelListContainer.innerHTML = '';

      if (models.length === 0) {
        modelListContainer.innerHTML = '<div class="model-empty-state"><span>No models in /jetson/models/</span></div>';
        return;
      }

      models.forEach(model => {
        const isActive = model.filename === activeModel;
        if (isActive) {
          window._currentModelClasses = model.classes;
        }
        const item = document.createElement('div');
        item.className = `model-item ${isActive ? 'active' : ''}`;

        const header = document.createElement('div');
        header.className = 'model-header';
        header.innerHTML = `
          <div class="model-name">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
            </svg>
            ${model.filename}
          </div>
          <div class="model-meta">${model.size} &bull; ${model.classes.length} cls</div>`;
        item.appendChild(header);

        // Class pills
        const pills = document.createElement('div');
        pills.className = 'class-pills';
        model.classes.slice(0, 8).forEach(cls => {
          const bin = model.mapping[cls] !== undefined ? model.mapping[cls] : 0;
          const pill = document.createElement('span');
          pill.className = `class-pill bin-${bin}`;
          pill.textContent = `${cls}`;
          pill.title = `${cls} → Bin ${bin}`;
          pill.addEventListener('click', () => openMappingModal(model));
          pills.appendChild(pill);
        });
        if (model.classes.length > 8) {
          const more = document.createElement('span');
          more.className = 'class-pill bin-0';
          more.textContent = `+${model.classes.length - 8} more`;
          more.addEventListener('click', () => openMappingModal(model));
          pills.appendChild(more);
        }
        item.appendChild(pills);

        // Actions
        const actions = document.createElement('div');
        actions.className = 'model-actions';

        if (!isActive) {
          const btnSel = document.createElement('button');
          btnSel.className = 'btn-xs';
          btnSel.textContent = 'Set Active';
          btnSel.addEventListener('click', () => selectModel(model.filename));
          actions.appendChild(btnSel);
        } else {
          const activePill = document.createElement('span');
          activePill.style.cssText = 'font-size:10px;color:var(--accent-cyan);font-family:var(--font-mono);padding:2px 6px;border:1px solid rgba(0,212,255,0.3);border-radius:4px;';
          activePill.textContent = '● ACTIVE';
          actions.appendChild(activePill);
        }

        const btnEdit = document.createElement('button');
        btnEdit.className = 'btn-xs';
        btnEdit.textContent = 'Edit Mappings';
        btnEdit.addEventListener('click', () => openMappingModal(model));
        actions.appendChild(btnEdit);

        item.appendChild(actions);
        modelListContainer.appendChild(item);
      });
    } catch (e) { console.error('loadModels failed:', e); }
  }

  async function selectModel(filename) {
    try {
      const res = await fetch('/api/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: filename })
      });
      const data = await res.json();
      if (data.success) { loadModels(); showToast(`Model switched to ${filename}`, 'success'); }
    } catch (e) { showToast('Failed to switch model', 'error'); }
  }

  // ── 4. MAPPING MODAL ──────────────────────────────────────────────────────
  function openMappingModal(model) {
    editingModel = model.filename;
    modalModelName.textContent = model.filename;
    mappingList.innerHTML = '';
    model.classes.forEach(cls => {
      const bin = model.mapping[cls] !== undefined ? model.mapping[cls] : 0;
      const row = document.createElement('div');
      row.className = 'mapping-row';
      row.innerHTML = `
        <span>${cls}</span>
        <select class="mapping-select" data-class="${cls}">
          <option value="0" ${bin===0?'selected':''}>Skip / Ignore</option>
          <option value="1" ${bin===1?'selected':''}>Bin 1 (Left)</option>
          <option value="2" ${bin===2?'selected':''}>Bin 2 (Center)</option>
          <option value="3" ${bin===3?'selected':''}>Bin 3 (Right)</option>
        </select>`;
      mappingList.appendChild(row);
    });
    mappingModal.classList.add('open');
  }

  const closeModal = () => mappingModal.classList.remove('open');
  btnCloseModal.addEventListener('click', closeModal);
  if (btnCancelModal) btnCancelModal.addEventListener('click', closeModal);
  mappingModal.addEventListener('click', (e) => { if (e.target === mappingModal) closeModal(); });

  btnSaveMapping.addEventListener('click', async () => {
    const mapping = {};
    mappingList.querySelectorAll('select.mapping-select').forEach(sel => {
      mapping[sel.dataset.class] = parseInt(sel.value, 10);
    });
    try {
      const res = await fetch('/api/models/mapping', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: editingModel, mapping })
      });
      const data = await res.json();
      if (data.success) { closeModal(); loadModels(); showToast('Mappings saved', 'success'); }
    } catch (e) { showToast('Failed to save mappings', 'error'); }
  });

  // ── 5. ARM COORDINATES ────────────────────────────────────────────────────
  async function loadCoordinates() {
    try {
      const res = await fetch('/api/arm/coordinates');
      const data = await res.json();
      if (!data.coordinates) return;
      const c = data.coordinates;
      const fill = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined) el.value = val; };
      fill('coord_pick_x', c.pick?.x); fill('coord_pick_y', c.pick?.y); fill('coord_pick_z', c.pick?.z);
      fill('coord_bin1_x', c.bin1?.x); fill('coord_bin1_y', c.bin1?.y); fill('coord_bin1_z', c.bin1?.z);
      fill('coord_bin2_x', c.bin2?.x); fill('coord_bin2_y', c.bin2?.y); fill('coord_bin2_z', c.bin2?.z);
      fill('coord_bin3_x', c.bin3?.x); fill('coord_bin3_y', c.bin3?.y); fill('coord_bin3_z', c.bin3?.z);
    } catch (e) { console.warn('loadCoordinates failed'); }
  }

  if (btnSaveCoords) {
    btnSaveCoords.addEventListener('click', async () => {
      const gv = (id) => parseFloat(document.getElementById(id)?.value ?? 0);
      const coords = {
        pick: { x: gv('coord_pick_x'), y: gv('coord_pick_y'), z: gv('coord_pick_z') },
        bin1: { x: gv('coord_bin1_x'), y: gv('coord_bin1_y'), z: gv('coord_bin1_z') },
        bin2: { x: gv('coord_bin2_x'), y: gv('coord_bin2_y'), z: gv('coord_bin2_z') },
        bin3: { x: gv('coord_bin3_x'), y: gv('coord_bin3_y'), z: gv('coord_bin3_z') },
      };
      try {
        const res = await fetch('/api/arm/coordinates', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ coordinates: coords })
        });
        const data = await res.json();
        if (data.success) showToast('Coordinates saved', 'success');
      } catch (e) { showToast('Failed to save coordinates', 'error'); }
    });
  }

  // ── 6. PIPELINE CONTROLS ──────────────────────────────────────────────────
  mockArmSwitch.addEventListener('change', async (e) => {
    try {
      await fetch('/api/pipeline/toggle_mock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mock: e.target.checked })
      });
      showToast(`Mock Arm ${e.target.checked ? 'ON' : 'OFF (real hardware)'}`, 'info');
    } catch (e) { console.warn('toggle_mock failed'); }
  });

  btnStartSorting.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/pipeline/start', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setPipelineRunningUI(true);
        updatePreviewUI(true);
        showToast('Sorting pipeline started!', 'success');
      } else {
        showToast('Could not start: ' + (data.error || 'Unknown'), 'error');
      }
    } catch (e) { showToast('Failed to start pipeline', 'error'); }
  });

  btnStopSorting.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/pipeline/stop', { method: 'POST' });
      const data = await res.json();
      if (data.success) { setPipelineRunningUI(false); showToast('Pipeline stopped. Arm returning home.', 'info'); }
    } catch (e) { showToast('Failed to stop pipeline', 'error'); }
  });

  function setPipelineRunningUI(running) {
    isPipelineRunning = running;
    if (running) {
      pipelineStatusBanner.innerHTML = '<span>&#9679; RUNNING</span>';
      pipelineStatusBanner.className = 'pipeline-status-banner status-running';
      btnStartSorting.disabled = true;
      btnStopSorting.disabled = false;
    } else {
      pipelineStatusBanner.innerHTML = '<span>&#9679; STOPPED</span>';
      pipelineStatusBanner.className = 'pipeline-status-banner status-stopped';
      btnStartSorting.disabled = false;
      btnStopSorting.disabled = true;
    }
  }

  // ── 7. LOG TERMINAL ───────────────────────────────────────────────────────
  async function pollLogs() {
    try {
      const res = await fetch('/api/logs?since=' + encodeURIComponent(lastLogTimestamp));
      const data = await res.json();

      if (data.logs && data.logs.length > 0) {
        data.logs.forEach(log => {
          lastLogTimestamp = log.timestamp;
          logEntryCount++;

          let cls = 'log-info';
          if (log.message.includes('[WARN]') || log.message.includes('[WARNING]')) cls = 'log-warn';
          else if (log.message.includes('[ERROR]') || log.message.includes('FAILED')) cls = 'log-error';
          else if (log.message.includes('[SUCCESS]') || log.message.includes('SUCCESS') || log.message.includes('Sorted')) cls = 'log-success';

          // Track sorted count from SUCCESS log messages
          if (cls === 'log-success' && log.message.includes('Sorted')) {
            totalSorted++;
            const binMatch = log.message.match(/Bin (\d)/);
            if (binMatch) {
              const b = parseInt(binMatch[1]);
              if (b >= 1 && b <= 3) binCounts[b]++;
            }
            updateBinMeters();
          }

          // ── Trigger arm simulation on CONFIRMED detection or sort cycle ───
          if (log.message.includes('[CONFIRMED]') || log.message.includes('Triggering Arm Cycle') || log.message.includes('[SUCCESS]') || log.message.includes('Sorted')) {
            const classMatch = log.message.match(/Detected\s+['"]?([^'"()]+)['"]?/i)
                            || log.message.match(/Sorted\s+['"]?([^'"()]+)['"]?/i)
                            || log.message.match(/class[:\s]+['"]?([\w\-]+)['"]?/i)
                            || log.message.match(/Sorting\s+([\w\-]+)/i)
                            || log.message.match(/([\w\-]+)\s+detected/i);
            const binMatch2 = log.message.match(/[Bb]in\s*(\d)/);
            if (classMatch && binMatch2) {
              const objClass = classMatch[1].trim();
              const binId = parseInt(binMatch2[1]);
              if (binId >= 1 && binId <= 3 && window._armSim) {
                window._armSim.triggerSort(objClass, binId);
              }
            }
          }

          const line = document.createElement('div');
          line.className = 'log-line';
          line.innerHTML = `
            <span class="log-prefix">&gt;</span>
            <span class="log-time">[${log.time}]</span>
            <span class="${cls}">${log.message}</span>`;
          terminalBox.appendChild(line);
        });
        terminalBox.scrollTop = terminalBox.scrollHeight;
        if (terminalLogCount) terminalLogCount.textContent = `${logEntryCount} entr${logEntryCount !== 1 ? 'ies' : 'y'}`;
      }

      // Sync state
      if (data.pipeline_running !== undefined && data.pipeline_running !== isPipelineRunning) {
        setPipelineRunningUI(data.pipeline_running);
      }
      if (data.mock_arm !== undefined && mockArmSwitch.checked !== data.mock_arm) {
        mockArmSwitch.checked = data.mock_arm;
      }
    } catch (e) { /* ignore on unload */ }
  }

  btnClearLogs.addEventListener('click', async () => {
    try {
      await fetch('/api/logs/clear', { method: 'POST' });
      terminalBox.innerHTML = '<div class="log-line"><span class="log-prefix">&gt;</span> <span class="log-info">[Terminal cleared]</span></div>';
      logEntryCount = 0;
      if (terminalLogCount) terminalLogCount.textContent = '0 entries';
    } catch (e) { console.warn('Clear logs failed'); }
  });

  // ── 7. INTERACTIVE HIWONDER JETARM OVERLAY & MOUSE-DRAG ON LIVE CAMERA ───
  function initDragSimulation() {
    const dragCanvas = document.getElementById('dragOverlayCanvas');
    const dragHintBadge = document.getElementById('dragHintBadge');
    const liveArmStatus = document.getElementById('liveArmStatus');
    const btnLiveCamDemo = document.getElementById('btnLiveCamDemo');
    if (!dragCanvas) return;

    const ctx = dragCanvas.getContext('2d');
    let activeDetection = null;
    let isDragging = false;
    let dragCurrentPos = { x: 0, y: 0 };
    let dragOffset = { x: 0, y: 0 };
    let currentMode = 'roi';
    let binZones = {
      1: { name: "BIN 1 (LEFT)", rect: [30, 50, 160, 130], color: "#ff4d6d" },
      2: { name: "BIN 2 (CENTER)", rect: [240, 50, 160, 130], color: "#00e676" },
      3: { name: "BIN 3 (RIGHT)", rect: [450, 50, 160, 130], color: "#00d4ff" },
    };

    // Hiwonder JetArm Geometry & State
    const BASE_POS = [320, 466]; // Outer radius 18, highest point y=448 (below y=440 pickup zone)
    const HOME_POS = [510, 448]; // Parked horizontally right outside pickup zone
    const BIN_POSITIONS = {
      1: [110, 115],
      2: [320, 115],
      3: [530, 115]
    };
    const PICKUP_ROI = [220, 260, 200, 180];

    let armTipX = HOME_POS[0];
    let armTipY = HOME_POS[1];
    let armGrip = 0.0;       // 0.0 = OPEN, 1.0 = GRIPPED
    let armLift = 0.0;       // 0.0 = Table, 1.0 = Lifted
    let armState = 'IDLE';
    let armStatusText = 'IDLE';

    // Sequence execution state
    let sequence = [];
    let currentStepIdx = -1;
    let stepStartTime = 0;
    let stepDuration = 0.5;
    let stepFromPos = [...HOME_POS];
    let stepToPos = [...HOME_POS];
    let isRunningSequence = false;
    let carriedBox = null;
    let pendingObject = null;

    function solveIK(targetX, targetY) {
      const bx = BASE_POS[0], by = BASE_POS[1];
      const dx = targetX - bx;
      const dy = targetY - by;
      const dist = Math.hypot(dx, dy) || 1.0;
      const mx = (bx + targetX) * 0.5;
      const my = (by + targetY) * 0.5;
      const px = -dy / dist;
      const py = dx / dist;
      const bend = Math.min(42.0, dist * 0.26);
      const elbowX = mx + px * bend;
      const elbowY = my + py * bend;
      return { elbow: [elbowX, elbowY], wrist: [targetX, targetY] };
    }

    function smoothEase(t) {
      return t < 0.5 ? 2.0 * t * t : -1.0 + (4.0 - 2.0 * t) * t;
    }

    function startSequenceStep(idx) {
      if (idx >= sequence.length) {
        isRunningSequence = false;
        armState = 'IDLE';
        armStatusText = 'IDLE';
        if (liveArmStatus) liveArmStatus.textContent = 'IDLE';
        return;
      }
      const step = sequence[idx];
      armState = step.name;
      armStatusText = step.status || step.name;
      if (liveArmStatus) liveArmStatus.textContent = armStatusText;
      stepFromPos = [armTipX, armTipY];
      stepToPos = [...step.to];
      stepDuration = step.dur;
      stepStartTime = performance.now() / 1000.0;
    }

    function triggerLiveDemoSort(targetBin, targetClass, targetCenter) {
      if (isRunningSequence || isDragging) return;

      const pickPos = targetCenter || [320, 350];
      const binId = targetBin || 1;
      const cls = targetClass || 'Cube';
      const binPos = BIN_POSITIONS[binId] || BIN_POSITIONS[2];

      pendingObject = { class_name: cls, bin_id: binId, start_time: performance.now() / 1000.0 };

      sequence = [
        { name: "TO_PICK", to: [pickPos[0], pickPos[1] - 25], dur: 0.45, lift: 0.4, grip: 0.0, status: "APPROACHING OBJECT" },
        { name: "DESCEND", to: [pickPos[0], pickPos[1]], dur: 0.28, lift: 0.0, grip: 0.0, status: "DESCENDING TO OBJECT" },
        { name: "GRIP", to: [pickPos[0], pickPos[1]], dur: 0.25, lift: 0.0, grip: 1.0, status: "GRIPPING OBJECT" },
        { name: "ASCEND", to: [pickPos[0], pickPos[1] - 30], dur: 0.30, lift: 0.8, grip: 1.0, status: "LIFTING OBJECT" },
        { name: "TO_BIN", to: [binPos[0], binPos[1] - 20], dur: 0.55, lift: 0.8, grip: 1.0, status: `SWEEPING TO BIN ${binId}` },
        { name: "DESCEND_BIN", to: [binPos[0], binPos[1]], dur: 0.25, lift: 0.2, grip: 1.0, status: `LOWERING INTO BIN ${binId}` },
        { name: "RELEASE", to: [binPos[0], binPos[1]], dur: 0.25, lift: 0.2, grip: 0.0, status: `RELEASING INTO BIN ${binId}` },
        { name: "ASCEND_BIN", to: [binPos[0], binPos[1] - 30], dur: 0.25, lift: 0.6, grip: 0.0, status: "CLEARING BIN" },
        { name: "TO_HOME", to: [...HOME_POS], dur: 0.50, lift: 0.0, grip: 0.0, status: "RETURNING HOME" },
      ];

      currentStepIdx = 0;
      isRunningSequence = true;
      startSequenceStep(0);
      showToast(`JetArm Demo Sort: ${cls} → Bin ${binId}`, 'info');
    }

    if (btnLiveCamDemo) {
      btnLiveCamDemo.addEventListener('click', () => {
        let bId = 1;
        let cName = 'Rubiks-Cube';
        let center = [320, 350];
        if (activeDetection && activeDetection.center) {
          center = activeDetection.center;
          bId = activeDetection.bin_id || 1;
          cName = activeDetection.class_name || 'Object';
        } else if (window._currentModelClasses && window._currentModelClasses.length > 0) {
          cName = window._currentModelClasses[0];
        }
        triggerLiveDemoSort(bId, cName, center);
      });
    }

    function getCanvasCoords(e) {
      const rect = dragCanvas.getBoundingClientRect();
      const scaleX = dragCanvas.width / (rect.width || 1);
      const scaleY = dragCanvas.height / (rect.height || 1);
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
      };
    }

    // Periodically query detection from backend
    async function pollCurrentDetection() {
      if (!isPreviewRunning) {
        return;
      }
      try {
        const res = await fetch('/api/detection/current');
        const data = await res.json();
        currentMode = data.mode || 'roi';
        if (data.bin_zones) {
          for (const [k, v] of Object.entries(data.bin_zones)) {
            if (binZones[k]) binZones[k].rect = v.rect;
          }
        }
        if (currentMode === 'roi') {
          if (dragHintBadge) dragHintBadge.style.display = 'inline-block';
          dragCanvas.style.pointerEvents = 'auto';
          if (!isDragging && !isRunningSequence) {
            activeDetection = data.detection;
          }
        } else {
          if (dragHintBadge) dragHintBadge.style.display = 'none';
          activeDetection = null;
        }
        if (data.arm_state && liveArmStatus && !isDragging) {
          liveArmStatus.textContent = data.arm_state;
        }
      } catch (e) {}
    }
    setInterval(pollCurrentDetection, 120);

    // ── Mouse Drag Events ───────────────────────────────────────────────────
    dragCanvas.addEventListener('mousedown', (e) => {
      if (currentMode !== 'roi' || isRunningSequence) return;
      const { x, y } = getCanvasCoords(e);

      let targetBox = null;
      if (activeDetection && activeDetection.bounding_box) {
        const [bx, by, bw, bh] = activeDetection.bounding_box;
        if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
          targetBox = { bx, by, bw, bh, class_name: activeDetection.class_name, confidence: activeDetection.confidence };
        }
      }

      if (targetBox) {
        isDragging = true;
        carriedBox = targetBox;
        dragOffset = { x: x - targetBox.bx, y: y - targetBox.by };
        dragCurrentPos = { x, y };
        dragCanvas.style.cursor = 'grabbing';
        armState = 'TRACK_DRAG';
        armGrip = 1.0;
        armLift = 0.4;
        armStatusText = `MANUAL SORT: ${targetBox.class_name.toUpperCase()}`;
        if (liveArmStatus) liveArmStatus.textContent = armStatusText;
      }
    });

    dragCanvas.addEventListener('mousemove', (e) => {
      if (currentMode !== 'roi') return;
      const { x, y } = getCanvasCoords(e);

      if (!isDragging) {
        if (!isRunningSequence && activeDetection && activeDetection.bounding_box) {
          const [bx, by, bw, bh] = activeDetection.bounding_box;
          if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
            dragCanvas.style.cursor = 'grab';
          } else {
            dragCanvas.style.cursor = 'default';
          }
        }
      } else {
        dragCanvas.style.cursor = 'grabbing';
        dragCurrentPos = { x, y };
        const dragX = x - dragOffset.x;
        const dragY = y - dragOffset.y;
        const curCx = dragX + carriedBox.bw / 2;
        const curCy = dragY + carriedBox.bh / 2;

        // Arm wrist tip smoothly follows dragged object
        armTipX = curCx;
        armTipY = curCy;
      }
    });

    window.addEventListener('mouseup', async (e) => {
      if (!isDragging) return;
      isDragging = false;
      dragCanvas.style.cursor = 'default';

      const { x, y } = getCanvasCoords(e);
      const curCx = (x - dragOffset.x) + carriedBox.bw / 2;
      const curCy = (y - dragOffset.y) + carriedBox.bh / 2;

      let droppedBin = null;
      for (const [binId, binInfo] of Object.entries(binZones)) {
        const [rx, ry, rw, rh] = binInfo.rect;
        if (curCx >= rx && curCx <= rx + rw && curCy >= ry && curCy <= ry + rh) {
          droppedBin = parseInt(binId);
          break;
        }
      }

      const cls = carriedBox.class_name;
      const conf = carriedBox.confidence;
      carriedBox = null;

      if (droppedBin !== null) {
        // Successful deposit into bin
        totalSorted += 1;
        binCounts[droppedBin] = (binCounts[droppedBin] || 0) + 1;
        updateBinMeters();

        if (window._armSim) {
          window._armSim.triggerSort(cls, droppedBin);
        }

        showToast(`✓ Sorted '${cls}' → Bin ${droppedBin} (Simulated)`, 'success');

        try {
          await fetch('/api/simulated_sort', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ class_name: cls, bin_id: droppedBin, confidence: conf })
          });
        } catch (err) {
          console.warn('simulated_sort report failed:', err);
        }
      } else {
        showToast('Dropped outside bins — returned to home', 'info');
      }

      // Smoothly return arm to Home
      armGrip = 0.0;
      armLift = 0.0;
      sequence = [
        { name: "TO_HOME", to: [...HOME_POS], dur: 0.50, lift: 0.0, grip: 0.0, status: "RETURNING HOME" }
      ];
      currentStepIdx = 0;
      isRunningSequence = true;
      startSequenceStep(0);
    });

    // ── High-Fidelity 60 FPS Render & Animation Loop ────────────────────────
    function renderLoop() {
      requestAnimationFrame(renderLoop);

      // Clear canvas
      ctx.clearRect(0, 0, dragCanvas.width, dragCanvas.height);

      if (!isPreviewRunning && !isRunningSequence) {
        return;
      }

      const now = performance.now() / 1000.0;

      // Update sequence animation
      if (isRunningSequence && currentStepIdx >= 0 && currentStepIdx < sequence.length) {
        const step = sequence[currentStepIdx];
        const elapsed = now - stepStartTime;
        const progress = Math.min(1.0, elapsed / Math.max(0.001, stepDuration));

        if (progress >= 1.0) {
          armTipX = stepToPos[0];
          armTipY = stepToPos[1];
          armGrip = step.grip !== undefined ? step.grip : armGrip;
          armLift = step.lift !== undefined ? step.lift : armLift;

          if (step.name === "GRIP") {
            carriedBox = pendingObject;
          } else if (step.name === "RELEASE") {
            if (carriedBox) {
              const droppedBin = carriedBox.bin_id || 1;
              const cls = carriedBox.class_name || 'Cube';
              totalSorted += 1;
              binCounts[droppedBin] = (binCounts[droppedBin] || 0) + 1;
              updateBinMeters();
              if (window._armSim) window._armSim.triggerSort(cls, droppedBin);
              fetch('/api/simulated_sort', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ class_name: cls, bin_id: droppedBin, confidence: 0.95 })
              }).catch(() => {});
            }
            carriedBox = null;
          }

          currentStepIdx += 1;
          startSequenceStep(currentStepIdx);
        } else {
          const ease = smoothEase(progress);
          armTipX = stepFromPos[0] + (stepToPos[0] - stepFromPos[0]) * ease;
          armTipY = stepFromPos[1] + (stepToPos[1] - stepFromPos[1]) * ease;

          if (step.name === "GRIP") armGrip = ease;
          else if (step.name === "RELEASE") armGrip = 1.0 - ease;

          if (step.name === "ASCEND") armLift = ease;
          else if (step.name === "DESCEND_BIN") armLift = 1.0 - ease * 0.8;
          else if (step.name === "ASCEND_BIN") armLift = 0.2 + ease * 0.6;
          else if (step.name === "TO_HOME") armLift = Math.max(0.0, 1.0 - ease * 1.5);
        }
      }

      // If dragging an object, draw the dragged box & guide line
      if (isDragging && carriedBox) {
        const [bx, by, bw, bh] = activeDetection ? activeDetection.bounding_box : [carriedBox.bx, carriedBox.by, carriedBox.bw, carriedBox.bh];
        const origCx = bx + bw / 2;
        const origCy = by + bh / 2;
        const dragX = dragCurrentPos.x - dragOffset.x;
        const dragY = dragCurrentPos.y - dragOffset.y;
        const curCx = dragX + bw / 2;
        const curCy = dragY + bh / 2;

        // 1. Connecting guide line
        ctx.save();
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.75)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(origCx, origCy);
        ctx.lineTo(curCx, curCy);
        ctx.stroke();
        ctx.restore();

        // 2. Dragged box highlight
        ctx.save();
        ctx.fillStyle = 'rgba(0, 212, 255, 0.22)';
        ctx.fillRect(dragX, dragY, bw, bh);
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = '#00ffff';
        ctx.shadowBlur = 14;
        ctx.strokeRect(dragX, dragY, bw, bh);

        // Reticle
        ctx.strokeStyle = '#ffff00';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(curCx - 8, curCy); ctx.lineTo(curCx + 8, curCy);
        ctx.moveTo(curCx, curCy - 8); ctx.lineTo(curCx, curCy + 8);
        ctx.stroke();

        // Drag badge
        const badgeText = `DRAGGING: ${carriedBox.class_name.toUpperCase()}`;
        ctx.font = 'bold 11px monospace';
        const tw = ctx.measureText(badgeText).width;
        ctx.fillStyle = 'rgba(6, 14, 24, 0.9)';
        ctx.fillRect(dragX, dragY - 22, tw + 16, 20);
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 1;
        ctx.strokeRect(dragX, dragY - 22, tw + 16, 20);
        ctx.fillStyle = '#00ffff';
        ctx.fillText(badgeText, dragX + 8, dragY - 8);

        // Highlight active bin
        for (const [binId, binInfo] of Object.entries(binZones)) {
          const [rx, ry, rw, rh] = binInfo.rect;
          if (curCx >= rx && curCx <= rx + rw && curCy >= ry && curCy <= ry + rh) {
            ctx.fillStyle = 'rgba(0, 230, 118, 0.30)';
            ctx.fillRect(rx, ry, rw, rh);
            ctx.strokeStyle = '#00e676';
            ctx.lineWidth = 3;
            ctx.shadowColor = '#00e676';
            ctx.shadowBlur = 18;
            ctx.strokeRect(rx, ry, rw, rh);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText(`RELEASE TO SORT -> BIN ${binId}`, rx + 16, ry + rh - 18);
          }
        }
        ctx.restore();
      }

      // If in ROI mode and not dragging, draw subtle pulsing reticle on detected object to invite dragging
      if (currentMode === 'roi' && !isDragging && !isRunningSequence && activeDetection && activeDetection.bounding_box) {
        const [bx, by, bw, bh] = activeDetection.bounding_box;
        ctx.save();
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.7)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(bx - 2, by - 2, bw + 4, bh + 4);
        ctx.restore();
      }

      // ── DRAW HIWONDER JETARM ────────────────────────────────────────────────
      // When live video preview is active, the JetArm is rendered directly inside
      // the authentic OpenCV video stream. Only draw canvas arm if video is stopped.
      if (!isPreviewRunning) {
        drawHiwonderJetArm(ctx, armTipX, armTipY, armGrip, armLift, armStatusText);
      }
    }

    function drawHiwonderJetArm(ctx, tipX, tipY, gripProg, lift, statusText) {
      const bx = BASE_POS[0], by = BASE_POS[1];
      const { elbow, wrist } = solveIK(tipX, tipY);
      const [ex, ey] = elbow;
      const [wx, wy] = wrist;

      // 1. Reach boundary arc & Bin Guide Lines
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.18)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(bx, by, 385, Math.PI * 1.15, Math.PI * 1.85);
      ctx.stroke();

      // Bin connector guide rays (starting strictly outside pickup zone)
      const [rx, ry, rw, rh] = PICKUP_ROI;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.20)';
      ctx.beginPath();
      ctx.moveTo(rx - 4, ry + 25); ctx.lineTo(BIN_POSITIONS[1][0], BIN_POSITIONS[1][1]);
      ctx.moveTo(320, ry - 4); ctx.lineTo(BIN_POSITIONS[2][0], BIN_POSITIONS[2][1]);
      ctx.moveTo(rx + rw + 4, ry + 25); ctx.lineTo(BIN_POSITIONS[3][0], BIN_POSITIONS[3][1]);
      ctx.stroke();
      ctx.restore();

      // 2. 3D Elevation Drop Shadow
      if (lift > 0.05) {
        const shOffX = 14 * lift;
        const shOffY = 16 * lift;
        ctx.save();
        ctx.strokeStyle = 'rgba(6, 8, 12, 0.55)';
        ctx.lineWidth = 8;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(bx + shOffX, by + shOffY);
        ctx.lineTo(ex + shOffX, ey + shOffY);
        ctx.lineTo(wx + shOffX, wy + shOffY);
        ctx.stroke();

        ctx.fillStyle = 'rgba(6, 8, 12, 0.55)';
        ctx.beginPath();
        ctx.ellipse(wx + shOffX, wy + shOffY, 18 * (1 + lift * 0.3), 8, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // 3. Upper Arm: Dual CNC Parallel Aluminum Beams (Base -> Elbow)
      const dx1 = ex - bx;
      const dy1 = ey - by;
      const len1 = Math.hypot(dx1, dy1) || 1.0;
      const nx1 = -dy1 / len1;
      const ny1 = dx1 / len1;
      const beamOffset = 6.5;

      const p1_a = [bx + nx1 * beamOffset, by + ny1 * beamOffset];
      const p1_b = [ex + nx1 * beamOffset, ey + ny1 * beamOffset];
      const p2_a = [bx - nx1 * beamOffset, by - ny1 * beamOffset];
      const p2_b = [ex - nx1 * beamOffset, ey - ny1 * beamOffset];

      // Dual outer dark slate beams
      ctx.save();
      ctx.strokeStyle = '#28323c';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(p1_a[0], p1_a[1]); ctx.lineTo(p1_b[0], p1_b[1]);
      ctx.moveTo(p2_a[0], p2_a[1]); ctx.lineTo(p2_b[0], p2_b[1]);
      ctx.stroke();

      // Inner metallic highlight
      ctx.strokeStyle = '#607080';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(p1_a[0], p1_a[1]); ctx.lineTo(p1_b[0], p1_b[1]);
      ctx.moveTo(p2_a[0], p2_a[1]); ctx.lineTo(p2_b[0], p2_b[1]);
      ctx.stroke();

      // Cross-truss bracing struts (skeleton cutouts)
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 1.5;
      for (const t of [0.33, 0.66]) {
        const cx1 = bx + dx1 * t;
        const cy1 = by + dy1 * t;
        ctx.beginPath();
        ctx.moveTo(cx1 - nx1 * beamOffset, cy1 - ny1 * beamOffset);
        ctx.lineTo(cx1 + nx1 * beamOffset, cy1 + ny1 * beamOffset);
        ctx.stroke();
      }

      // Center cyan glow spine
      ctx.strokeStyle = 'rgba(0, 212, 255, 0.8)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(bx, by); ctx.lineTo(ex, ey);
      ctx.stroke();
      ctx.restore();

      // 4. Forearm Beam: Elbow -> Wrist
      const dx2 = wx - ex;
      const dy2 = wy - ey;
      const len2 = Math.hypot(dx2, dy2) || 1.0;
      const nx2 = -dy2 / len2;
      const ny2 = dx2 / len2;

      ctx.save();
      // Main forearm structural body
      ctx.strokeStyle = '#28323c';
      ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.moveTo(ex, ey); ctx.lineTo(wx, wy);
      ctx.stroke();

      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ex, ey); ctx.lineTo(wx, wy);
      ctx.stroke();

      // External pushrod linkage
      const rodOff = 5.0;
      ctx.strokeStyle = '#a0b4c8';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(ex + nx2 * rodOff, ey + ny2 * rodOff);
      ctx.lineTo(wx + nx2 * rodOff, wy + ny2 * rodOff);
      ctx.stroke();
      ctx.restore();

      // 5. Elbow Joint Disc (J2 / J3)
      ctx.save();
      ctx.fillStyle = '#181e26';
      ctx.strokeStyle = '#ffd400';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(ex, ey, 11, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Servo horn bolts
      ctx.fillStyle = '#b4b4b4';
      for (let ang = 0; ang < 360; ang += 90) {
        const rad = ang * Math.PI / 180;
        ctx.beginPath();
        ctx.arc(ex + Math.cos(rad) * 6.5, ey + Math.sin(rad) * 6.5, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
      // Central silver pivot pin
      ctx.fillStyle = '#d0d8e0';
      ctx.beginPath();
      ctx.arc(ex, ey, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // 6. Base Turntable Platform (Waist Joint J1)
      ctx.save();
      // Outer heavy circular base (radius 18)
      ctx.fillStyle = '#161c24';
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(bx, by, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // 4 perimeter socket bolts
      ctx.fillStyle = '#b4b4b4';
      for (let ang = 45; ang < 360; ang += 90) {
        const rad = ang * Math.PI / 180;
        ctx.beginPath();
        ctx.arc(bx + Math.cos(rad) * 13, by + Math.sin(rad) * 13, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }

      // Inner rotary plate
      ctx.fillStyle = '#0e1218';
      ctx.strokeStyle = '#506478';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(bx, by, 11, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Core cyan LED ring & central brass hub
      ctx.fillStyle = '#00d4ff';
      ctx.beginPath();
      ctx.arc(bx, by, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(bx, by, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // 7. Wrist Pivot & Hiwonder Twin-Jaw Parallel Gripper
      ctx.save();
      ctx.fillStyle = '#181e26';
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(wx, wy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      const dirX = dx2 / len2;
      const dirY = dy2 / len2;
      const perpX = -dirY;
      const perpY = dirX;

      const spread = 15.0 - (gripProg * 11.0);
      const jawLen = 16.0;
      const jawCol = gripProg > 0.5 ? '#ffd400' : '#00e676';

      for (const side of [-1, 1]) {
        const rootX = wx + perpX * spread * side;
        const rootY = wy + perpY * spread * side;
        const midX = rootX + dirX * (jawLen * 0.6) + perpX * (side * 2.0);
        const midY = rootY + dirY * (jawLen * 0.6) + perpY * (side * 2.0);
        const tipFX = midX + dirX * (jawLen * 0.5) - perpX * (side * 3.5);
        const tipFY = midY + dirY * (jawLen * 0.5) - perpY * (side * 3.5);

        ctx.strokeStyle = jawCol;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(rootX, rootY);
        ctx.lineTo(midX, midY);
        ctx.lineTo(tipFX, tipFY);
        ctx.stroke();

        // Rubber grip pad dot
        ctx.fillStyle = '#282828';
        ctx.beginPath();
        ctx.arc(tipFX, tipFY, 2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Gripper state badge: [ OPEN ] or [ GRIP ]
      const gripBadgeText = gripProg > 0.5 ? '[ GRIP ]' : '[ OPEN ]';
      ctx.font = 'bold 9px monospace';
      ctx.fillStyle = gripProg > 0.5 ? 'rgba(255, 212, 0, 0.9)' : 'rgba(0, 230, 118, 0.9)';
      ctx.fillText(gripBadgeText, wx - 18, wy + 20);
      ctx.restore();
    }

    // Start 60 FPS animation loop
    requestAnimationFrame(renderLoop);
  }

  // ── Initial Boot ──────────────────────────────────────────────────────────
  loadModels();
  loadCoordinates();
  loadCameras();
  initDragSimulation();
  setInterval(pollLogs, 900);

  // ── Sync Live Cam Sorting Mode (Auto / Manual) ─────────────────────────────
  async function syncSortingMode() {
    try {
      const res = await fetch('/api/arm/mode');
      const data = await res.json();
      if (data.mode === 'auto') {
        if (btnModeAuto) btnModeAuto.classList.add('active');
        if (btnModeManual) btnModeManual.classList.remove('active');
      } else {
        if (btnModeManual) btnModeManual.classList.add('active');
        if (btnModeAuto) btnModeAuto.classList.remove('active');
      }
    } catch (e) {}
  }
  syncSortingMode();

  // ── Wire Live Camera Feed "▶ Demo Sort" Button ─────────────────────────────
  const btnLiveCamDemo = document.getElementById('btnLiveCamDemo');
  if (btnLiveCamDemo) {
    btnLiveCamDemo.addEventListener('click', async () => {
      let targetCls = 'Rubiks-Cube';
      let targetBin = 1;
      if (activeDetection && activeDetection.class_name) {
        targetCls = activeDetection.class_name;
        targetBin = activeDetection.bin_id || 1;
      }
      try {
        const res = await fetch('/api/simulated_sort', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ class_name: targetCls, bin_id: targetBin, confidence: 0.95 })
        });
        const data = await res.json();
        if (data.success) {
          showToast(`Demo Sort: ${targetCls} → Bin ${targetBin}`, 'success');
        }
      } catch (e) {
        showToast('Demo sort request failed', 'error');
      }
    });
  }

  // ── Auto-Start Live Video Preview On Dashboard Launch ──────────────────────
  setTimeout(() => {
    updatePreviewUI(true);
  }, 350);

  // ── Arm Simulator Init ────────────────────────────────────────────────────
  const simStatusBadge = document.getElementById('simStatusBadge');
  const simObjectLabel = document.getElementById('simObjectLabel');
  const btnSimDemo     = document.getElementById('btnSimDemo');

  // Init simulator after DOM is painted so canvas has layout dimensions
  requestAnimationFrame(() => {
    window._armSim = new ArmSimulator('armSimCanvas', simStatusBadge, simObjectLabel);
  });

  if (btnSimDemo) {
    btnSimDemo.addEventListener('click', () => {
      if (window._armSim) {
        let targetCls = 'Rubiks-Cube';
        let targetBin = 1;
        if (window._currentModelClasses && window._currentModelClasses.length > 0) {
          targetCls = window._currentModelClasses[0];
        }
        window._armSim.triggerSort(targetCls, targetBin);
        showToast(`Demo: ${targetCls} → Bin ${targetBin}`, 'info');
      }
    });
  }
});

// ==============================================================================
// ARM SIMULATOR — 2D Top-Down Canvas Animation
// Triggered by live log parsing or Demo button.
// ==============================================================================
class ArmSimulator {
  constructor(canvasId, statusBadge, objectLabelEl) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.statusBadge  = statusBadge;
    this.objectLabelEl = objectLabelEl;

    // Fix canvas pixel dimensions to match CSS layout size
    this._resize();
    window.addEventListener('resize', () => this._resize());

    const W = this.W, H = this.H;

    // Arm base position (bottom center)
    this.base = { x: W / 2, y: H - 28 };

    // Named target positions for arm tip
    this.poses = {
      home:  { x: W / 2,        y: H - 55 },
      pick:  { x: W / 2,        y: H * 0.38 },
      bin1:  { x: W * 0.80,     y: H * 0.32 },
      bin2:  { x: W / 2,        y: H * 0.18 },
      bin3:  { x: W * 0.20,     y: H * 0.32 },
    };
    this.pickCenter = { ...this.poses.pick };

    // Animation state
    this.tipPos        = { ...this.poses.home };
    this.fromPos       = { ...this.poses.home };
    this.toPos         = { ...this.poses.home };
    this.stepProgress  = 1;
    this.stepDuration  = 60;
    this.stepLift      = 0;
    this.stepGrip      = 0;
    this.liftAmount    = 0;
    this.gripProgress  = 0;  // 0=open 1=closed

    this.sequence      = [];
    this.currentStep   = -1;
    this.running       = false;
    this.sortCount     = 0;
    this.carryObject   = null;  // { label, color, bin }
    this.objectAtPickup = null; // { label, color }

    // Polyfill CanvasRenderingContext2D.roundRect for older browsers
    if (!CanvasRenderingContext2D.prototype.roundRect) {
      CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
        this.beginPath();
        this.moveTo(x + r, y);
        this.lineTo(x + w - r, y);
        this.arcTo(x + w, y, x + w, y + r, r);
        this.lineTo(x + w, y + h - r);
        this.arcTo(x + w, y + h, x + w - r, y + h, r);
        this.lineTo(x + r, y + h);
        this.arcTo(x, y + h, x, y + h - r, r);
        this.lineTo(x, y + r);
        this.arcTo(x, y, x + r, y, r);
        this.closePath();
        return this;
      };
    }

    this._frame();
  }

  _resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.W = rect.width  || 900;
    this.H = rect.height || 280;
    this.canvas.width  = this.W;
    this.canvas.height = this.H;
    // Recalculate layout-dependent positions
    const W = this.W, H = this.H;
    this.base      = { x: W / 2, y: H - 28 };
    this.poses = {
      home:  { x: W / 2,    y: H - 55 },
      pick:  { x: W / 2,    y: H * 0.38 },
      bin1:  { x: W * 0.80, y: H * 0.32 },
      bin2:  { x: W / 2,    y: H * 0.18 },
      bin3:  { x: W * 0.20, y: H * 0.32 },
    };
    if (this.poses) this.pickCenter = { ...this.poses.pick };
  }

  triggerSort(label, binId) {
    if (this.running) return;
    const binKey = `bin${binId}`;
    const tp = this.poses[binKey];
    if (!tp) return;

    const COLORS = { 1: '#00d4ff', 2: '#00e676', 3: '#ff9800' };
    const color = COLORS[binId] || '#fff';
    this.carryObject   = { label, color, bin: binId };
    this.objectAtPickup = { label, color };
    this.sortCount++;

    const hp = this.poses.home;
    const pp = this.poses.pick;

    this.sequence = [
      { name: 'TO_PICK',     from: hp,           to: pp,  dur: 80,  lift: 0, grip: 0 },
      { name: 'DESCEND',     from: pp,            to: pp,  dur: 28,  lift: 0, grip: 0 },
      { name: 'GRIP',        from: pp,            to: pp,  dur: 38,  lift: 0, grip: 1 },
      { name: 'ASCEND',      from: pp,            to: pp,  dur: 36,  lift: 1, grip: 1 },
      { name: 'TO_BIN',      from: pp,            to: tp,  dur: 100, lift: 1, grip: 1 },
      { name: 'DESCEND_BIN', from: tp,            to: tp,  dur: 32,  lift: 0, grip: 1 },
      { name: 'RELEASE',     from: tp,            to: tp,  dur: 38,  lift: 0, grip: 0 },
      { name: 'ASCEND_BIN',  from: tp,            to: tp,  dur: 28,  lift: 1, grip: 0 },
      { name: 'TO_HOME',     from: tp,            to: hp,  dur: 90,  lift: 0, grip: 0 },
    ];
    this.currentStep  = 0;
    this.running      = true;
    this._startStep(0);
  }

  _startStep(i) {
    if (i >= this.sequence.length) {
      this.running      = false;
      this.carryObject  = null;
      this.objectAtPickup = null;
      this.liftAmount   = 0;
      this.gripProgress = 0;
      if (this.statusBadge) { this.statusBadge.textContent = 'IDLE'; this.statusBadge.style.color = ''; }
      if (this.objectLabelEl) this.objectLabelEl.textContent = '';
      return;
    }
    const s = this.sequence[i];
    this.fromPos      = { x: this.tipPos.x, y: this.tipPos.y };
    this.toPos        = { ...s.to };
    this.stepDuration = s.dur;
    this.stepProgress = 0;
    this.stepLift     = s.lift;
    this.stepGrip     = s.grip;

    if (this.statusBadge) {
      const labels = {
        TO_PICK: '▶ MOVING TO PICK',  DESCEND: '▼ DESCENDING',
        GRIP: '⊙ GRIPPING',           ASCEND: '▲ LIFTING',
        TO_BIN: `▶ TO BIN ${this.carryObject?.bin}`,
        DESCEND_BIN: '▼ PLACING',     RELEASE: '○ RELEASING',
        ASCEND_BIN: '▲ CLEARING',     TO_HOME: '⌂ RETURNING HOME',
      };
      this.statusBadge.textContent = labels[s.name] || s.name;
      this.statusBadge.style.color = 'var(--accent-cyan)';
    }
    if (this.objectLabelEl && this.carryObject) {
      this.objectLabelEl.textContent = `${this.carryObject.label} → Bin ${this.carryObject.bin}`;
    }
  }

  _update() {
    if (!this.running) return;
    this.stepProgress += 1 / this.stepDuration;
    if (this.stepProgress >= 1) {
      this.stepProgress = 1;
      this.tipPos = { ...this.toPos };
      this.liftAmount   = 0;
      this.gripProgress = this.stepGrip;
      const name = this.sequence[this.currentStep]?.name;
      if (name === 'RELEASE')  this.carryObject   = null;
      if (name === 'GRIP')     this.objectAtPickup = null;
      this.currentStep++;
      this._startStep(this.currentStep);
    } else {
      const t = this._ease(this.stepProgress);
      this.tipPos = {
        x: this.fromPos.x + (this.toPos.x - this.fromPos.x) * t,
        y: this.fromPos.y + (this.toPos.y - this.fromPos.y) * t,
      };
      this.liftAmount   = this.stepLift > 0 ? Math.sin(this.stepProgress * Math.PI) : 0;
      this.gripProgress = this.stepGrip;
    }
  }

  _ease(t) { return t < 0.5 ? 2*t*t : -1 + (4 - 2*t)*t; }

  _frame() {
    this._update();
    this._draw();
    requestAnimationFrame(() => this._frame());
  }

  // ── Drawing ────────────────────────────────────────────────────────────────
  _draw() {
    const ctx = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Background
    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, '#02070f');
    bg.addColorStop(1, '#040c18');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    this._drawGrid(ctx, W, H);
    this._drawReachBoundary(ctx);
    this._drawBins(ctx, W);
    this._drawPickupZone(ctx);

    // Object at pickup (before gripped)
    if (this.objectAtPickup) {
      this._drawObject(ctx, this.pickCenter.x, this.pickCenter.y, this.objectAtPickup, 0);
    }

    this._drawArmShadow(ctx);
    this._drawArm(ctx);

    // Carried object follows arm tip
    if (this.carryObject && this.gripProgress > 0.5) {
      this._drawObject(ctx, this.tipPos.x, this.tipPos.y, this.carryObject, this.liftAmount);
    }

    this._drawOverlay(ctx, W, H);
  }

  _drawGrid(ctx, W, H) {
    ctx.strokeStyle = 'rgba(0,212,255,0.035)';
    ctx.lineWidth = 1;
    const gs = 40;
    for (let x = gs; x < W; x += gs) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = gs; y < H; y += gs) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
  }

  _drawReachBoundary(ctx) {
    const maxR = Math.min(this.W, this.H) * 0.70;
    ctx.save();
    ctx.strokeStyle = 'rgba(0,212,255,0.07)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 8]);
    ctx.beginPath();
    ctx.arc(this.base.x, this.base.y, maxR, Math.PI, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  _drawBins(ctx, W) {
    const BIN_DEF = [
      { key: 'bin1', label: 'BIN 1', color: '#00d4ff', w: 100, h: 46 },
      { key: 'bin2', label: 'BIN 2', color: '#00e676', w: 100, h: 42 },
      { key: 'bin3', label: 'BIN 3', color: '#ff9800', w: 100, h: 46 },
    ];
    BIN_DEF.forEach(b => {
      const pos = this.poses[b.key];
      const bx = pos.x - b.w / 2, by = pos.y - b.h / 2;

      // Fill
      ctx.fillStyle = b.color + '1a';
      ctx.beginPath();
      ctx.roundRect(bx, by, b.w, b.h, 7);
      ctx.fill();

      // Border
      ctx.strokeStyle = b.color + '80';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(bx, by, b.w, b.h, 7);
      ctx.stroke();

      // Top edge glow strip
      ctx.strokeStyle = b.color + 'cc';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(bx + 7, by); ctx.lineTo(bx + b.w - 7, by);
      ctx.stroke();

      // Label
      ctx.fillStyle = b.color;
      ctx.font = 'bold 12px \'Rajdhani\', sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(b.label, pos.x, pos.y);

      // Connector line from base to bin (faint guide)
      ctx.strokeStyle = b.color + '18';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 7]);
      ctx.beginPath();
      ctx.moveTo(this.base.x, this.base.y);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  }

  _drawPickupZone(ctx) {
    const p = this.pickCenter;
    const sz = 46;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,193,7,0.55)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(p.x - sz/2, p.y - sz/2, sz, sz);
    ctx.setLineDash([]);
    // Crosshair
    ctx.strokeStyle = 'rgba(255,193,7,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(p.x - 12, p.y); ctx.lineTo(p.x + 12, p.y);
    ctx.moveTo(p.x, p.y - 12); ctx.lineTo(p.x, p.y + 12);
    ctx.stroke();
    // Label
    ctx.fillStyle = 'rgba(255,193,7,0.65)';
    ctx.font = '9px \'JetBrains Mono\', monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('PICK ZONE', p.x, p.y + sz/2 + 4);
    ctx.restore();
  }

  _drawObject(ctx, x, y, obj, lift) {
    const sz = 22;
    const shadowX = lift * 14, shadowY = lift * 10;
    const floatY  = lift * 16;
    ctx.save();
    // Ground shadow when lifted
    if (lift > 0.05) {
      ctx.globalAlpha = 0.25 * lift;
      ctx.fillStyle = '#000';
      ctx.beginPath();
      ctx.ellipse(x + shadowX, y + shadowY + sz/2, sz * 0.55 * (1 + lift * 0.3), sz * 0.22, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    // Object square
    const ox = x - sz/2;
    const oy = y - sz/2 - floatY;
    ctx.fillStyle = obj.color;
    ctx.globalAlpha = 0.92;
    ctx.beginPath();
    ctx.roundRect(ox, oy, sz, sz, 5);
    ctx.fill();
    ctx.globalAlpha = 1;
    // Highlight edge
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(ox, oy, sz, sz, 5);
    ctx.stroke();
    // Inner text inside cube
    const shortLbl = obj.label.length > 5 ? obj.label.substring(0, 4) : obj.label;
    ctx.fillStyle = 'rgba(0,0,0,0.85)';
    ctx.font = 'bold 8px \'Inter\', sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(shortLbl.toUpperCase(), ox + sz/2, oy + sz/2);

    // Floating HUD Tag above object showing full label and bin
    const tagText = `${obj.label} → BIN ${obj.bin || '1'}`.trim();
    ctx.font = 'bold 10px \'JetBrains Mono\', monospace';
    const tw = ctx.measureText(tagText).width;
    const pw = tw + 14, ph = 18;
    const px = x - pw / 2, py = oy - ph - 6;

    // Badge background
    ctx.fillStyle = 'rgba(4, 14, 26, 0.92)';
    ctx.beginPath();
    ctx.roundRect(px, py, pw, ph, 4);
    ctx.fill();

    // Badge border
    ctx.strokeStyle = obj.color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(px, py, pw, ph, 4);
    ctx.stroke();

    // Badge text
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(tagText, x, py + ph / 2);
    ctx.restore();
  }

  _getElbow(base, tip) {
    const mx = (base.x + tip.x) / 2;
    const my = (base.y + tip.y) / 2;
    const dx = tip.x - base.x, dy = tip.y - base.y;
    const len = Math.sqrt(dx*dx + dy*dy) || 1;
    const px = -dy / len, py = dx / len;  // perpendicular
    const bend = Math.min(45, len * 0.28);
    return { x: mx + px * bend, y: my + py * bend };
  }

  _drawArmShadow(ctx) {
    if (this.liftAmount < 0.05) return;
    const off = this.liftAmount * 14;
    const bs = { x: this.base.x + off, y: this.base.y + off };
    const ts = { x: this.tipPos.x + off, y: this.tipPos.y + off };
    const es = this._getElbow(bs, ts);
    ctx.save();
    ctx.globalAlpha = 0.18 * this.liftAmount;
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 9;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(bs.x, bs.y);
    ctx.quadraticCurveTo(es.x, es.y, ts.x, ts.y);
    ctx.stroke();
    ctx.restore();
  }

  _drawArm(ctx) {
    const base = this.base;
    const tip  = this.tipPos;
    const elbow = this._getElbow(base, tip);
    const glow = 0.6 + this.liftAmount * 0.4;

    // Upper arm gradient
    const ag = ctx.createLinearGradient(base.x, base.y, elbow.x, elbow.y);
    ag.addColorStop(0, `rgba(0,160,200,${0.55*glow})`);
    ag.addColorStop(1, `rgba(0,212,255,${0.90*glow})`);

    // Outer arm body
    ctx.save();
    ctx.lineWidth = 9;
    ctx.lineCap = 'round';
    ctx.strokeStyle = ag;
    ctx.beginPath();
    ctx.moveTo(base.x, base.y);
    ctx.quadraticCurveTo(elbow.x, elbow.y, tip.x, tip.y);
    ctx.stroke();

    // Inner glow core
    ctx.lineWidth = 3;
    ctx.strokeStyle = `rgba(180,240,255,${0.35*glow})`;
    ctx.beginPath();
    ctx.moveTo(base.x, base.y);
    ctx.quadraticCurveTo(elbow.x, elbow.y, tip.x, tip.y);
    ctx.stroke();
    ctx.restore();

    // Base joint
    ctx.save();
    ctx.fillStyle = '#061a2a';
    ctx.strokeStyle = `rgba(0,212,255,${0.85*glow})`;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(base.x, base.y, 11, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = `rgba(0,212,255,${glow})`;
    ctx.beginPath(); ctx.arc(base.x, base.y, 4, 0, Math.PI*2); ctx.fill();
    ctx.restore();

    // Elbow joint
    ctx.save();
    ctx.fillStyle = '#061a2a';
    ctx.strokeStyle = `rgba(0,212,255,${0.55*glow})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(elbow.x, elbow.y, 5.5, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.restore();

    // Gripper
    this._drawGripper(ctx, tip, elbow, glow);
  }

  _drawGripper(ctx, tip, elbow, glow) {
    const dx = tip.x - elbow.x, dy = tip.y - elbow.y;
    const len = Math.sqrt(dx*dx + dy*dy) || 1;
    const dirX = dx/len, dirY = dy/len;
    const perpX = -dirY, perpY = dirX;
    const spread = (1 - this.gripProgress) * 9 + 2;
    const jawLen = 14;

    const jawColor = this.gripProgress > 0.5
      ? `rgba(0,212,255,${0.9*glow})`
      : `rgba(0,230,118,${0.85*glow})`;

    // Palm circle
    ctx.save();
    ctx.fillStyle = '#061a2a';
    ctx.strokeStyle = jawColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(tip.x, tip.y, 6.5, 0, Math.PI*2); ctx.fill(); ctx.stroke();
    ctx.restore();

    // Two jaws
    ctx.save();
    ctx.strokeStyle = jawColor;
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    [1, -1].forEach(side => {
      const jbx = tip.x + perpX * spread * side;
      const jby = tip.y + perpY * spread * side;
      ctx.beginPath();
      ctx.moveTo(jbx, jby);
      ctx.lineTo(jbx + dirX * jawLen, jby + dirY * jawLen);
      ctx.stroke();
    });
    ctx.restore();

    // Gripper label
    ctx.save();
    ctx.fillStyle = this.gripProgress > 0.5 ? 'rgba(0,212,255,0.6)' : 'rgba(0,230,118,0.6)';
    ctx.font = '8px \'JetBrains Mono\', monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(this.gripProgress > 0.5 ? 'CLOSED' : 'OPEN', tip.x, tip.y - 14);
    ctx.restore();
  }

  _drawOverlay(ctx, W, H) {
    const pad = 12;

    if (!this.running) {
      // Idle hint
      ctx.fillStyle = 'rgba(0,212,255,0.2)';
      ctx.font = '10px \'JetBrains Mono\', monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText('ARM: IDLE — AWAITING DETECTION', pad, H - 10);
      ctx.fillStyle = 'rgba(255,255,255,0.1)';
      ctx.fillText(`Cycles: ${this.sortCount}  |  Press "Demo" to preview`, pad, H - 24);
      return;
    }

    const step = this.sequence[this.currentStep];
    if (!step) return;

    // Step labels with icons
    const stepLabels = {
      TO_PICK: '▶ MOVING TO PICKUP ZONE',
      DESCEND: '▼ DESCENDING TO OBJECT',
      GRIP:    '⊙ GRIPPING OBJECT',
      ASCEND:  '▲ LIFTING OBJECT',
      TO_BIN:  `▶ CARRYING TO BIN ${this.carryObject?.bin || '?'}`,
      DESCEND_BIN: '▼ LOWERING INTO BIN',
      RELEASE: '○ RELEASING OBJECT',
      ASCEND_BIN: '▲ CLEARING BIN',
      TO_HOME: '⌂ RETURNING TO HOME',
    };

    // Current step label (bottom-left)
    ctx.save();
    ctx.fillStyle = 'rgba(0,212,255,0.95)';
    ctx.font = 'bold 12px \'Rajdhani\', sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(stepLabels[step.name] || step.name, pad, H - 10);

    // Object label
    if (this.carryObject) {
      ctx.fillStyle = this.carryObject.color;
      ctx.font = 'bold 11px \'JetBrains Mono\', monospace';
      ctx.fillText(`OBJECT: ${this.carryObject.label.toUpperCase()}  ▶  TARGET: BIN ${this.carryObject.bin}`, pad, H - 28);
    }

    // Step progress bar (bottom-right)
    const barW = 140, barH = 3;
    const barX = W - barW - pad, barY = H - 14;
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = 'rgba(0,212,255,0.7)';
    ctx.fillRect(barX, barY, barW * this.stepProgress, barH);

    // Step counter
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '9px \'JetBrains Mono\', monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`STEP ${this.currentStep + 1} / ${this.sequence.length}  ·  Cycle #${this.sortCount}`, W - pad, H - 18);

    ctx.restore();
  }
}
