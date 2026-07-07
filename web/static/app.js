/* CorridorKey Web UI — Main Application JS */

const state = {
    clips: [],
    selectedClip: null,
    previewFrame: 0,
    previewLayer: 'input',
    jobs: [],
    ws: null,
};

// --- Init ---

document.addEventListener('DOMContentLoaded', () => {
    fetchDevice();
    fetchClips();
    fetchJobs();
    connectWS();
});

// --- API ---

async function fetchDevice() {
    try {
        const res = await fetch('/api/device');
        const data = await res.json();
        const el = document.getElementById('device-name');
        if (data.vram && data.vram.name) {
            const free = data.vram.free ? data.vram.free.toFixed(1) : '?';
            el.textContent = `${data.vram.name} | ${free}GB free`;
        } else {
            el.textContent = data.device.toUpperCase();
        }
    } catch (e) {
        document.getElementById('device-name').textContent = 'offline';
    }
}

async function fetchClips() {
    try {
        const res = await fetch('/api/clips');
        state.clips = await res.json();
        renderClipList();
    } catch (e) {
        console.error('Failed to fetch clips:', e);
    }
}

async function scanClips() {
    const path = document.getElementById('path-input').value.trim();
    const body = path ? { path } : {};
    try {
        const btn = document.getElementById('scan-btn');
        btn.textContent = 'Scanning...';
        btn.disabled = true;
        const res = await fetch('/api/clips/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Scan failed', 'error');
            return;
        }
        state.clips = await res.json();
        renderClipList();
    } catch (e) {
        showToast('Scan failed: ' + e.message, 'error');
    } finally {
        const btn = document.getElementById('scan-btn');
        btn.textContent = 'Scan';
        btn.disabled = false;
    }
}

async function fetchJobs() {
    try {
        const res = await fetch('/api/jobs');
        state.jobs = await res.json();
        renderJobQueue();
    } catch (e) {
        console.error('Failed to fetch jobs:', e);
    }
}

async function submitInference(clipName) {
    const settings = getSettings();
    try {
        const res = await fetch(`/api/clips/${encodeURIComponent(clipName)}/inference`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings),
        });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Failed to queue', 'error');
            return;
        }
        fetchJobs();
    } catch (e) {
        showToast('Failed: ' + e.message, 'error');
    }
}

async function submitGVM(clipName) {
    try {
        const res = await fetch(`/api/clips/${encodeURIComponent(clipName)}/gvm`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Failed to queue', 'error');
            return;
        }
        fetchJobs();
    } catch (e) {
        showToast('Failed: ' + e.message, 'error');
    }
}

async function submitExtract(clipName) {
    try {
        const res = await fetch(`/api/clips/${encodeURIComponent(clipName)}/extract`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Failed to queue extraction', 'error');
            return;
        }
        fetchJobs();
    } catch (e) {
        showToast('Failed: ' + e.message, 'error');
    }
}

async function submitVideoMaMa(clipName) {
    try {
        const res = await fetch(`/api/clips/${encodeURIComponent(clipName)}/videomama`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chunk_size: 50 }),
        });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Failed to queue', 'error');
            return;
        }
        fetchJobs();
    } catch (e) {
        showToast('Failed: ' + e.message, 'error');
    }
}

async function cancelJob(jobId) {
    try {
        await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
        fetchJobs();
    } catch (e) {
        console.error('Cancel failed:', e);
    }
}

async function deleteClipOutputs(clipName) {
    if (!confirm(`Clear all outputs for "${clipName}"? Input files will be kept.`)) return;
    try {
        const res = await fetch('/api/clips/delete-outputs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: clipName }),
        });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Delete failed', 'error');
            return;
        }
        if (state.selectedClip === clipName) {
            state.selectedClip = null;
            renderPreview();
        }
        fetchClips();
        fetchJobs();
    } catch (e) {
        showToast('Delete failed: ' + e.message, 'error');
    }
}

async function deleteClipEntirely(clipName) {
    if (!confirm(`Permanently delete "${clipName}" and ALL its files? This cannot be undone.`)) return;
    try {
        const res = await fetch('/api/clips/delete-all', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: clipName }),
        });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Delete failed', 'error');
            return;
        }
        if (state.selectedClip === clipName) {
            state.selectedClip = null;
            renderPreview();
        }
        fetchClips();
        fetchJobs();
    } catch (e) {
        showToast('Delete failed: ' + e.message, 'error');
    }
}

// --- Settings ---

function getSettings() {
    const despillVal = parseInt(document.getElementById('set-despill').value, 10);
    return {
        input_is_linear: document.getElementById('set-colorspace').value === 'linear',
        despill_strength: despillVal / 10.0,
        auto_despeckle: document.getElementById('set-despeckle').checked,
        despeckle_size: parseInt(document.getElementById('set-despeckle-size').value, 10) || 400,
        refiner_scale: parseFloat(document.getElementById('set-refiner').value) || 1.0,
        fg_enabled: document.getElementById('out-fg').checked,
        fg_format: 'exr',
        matte_enabled: document.getElementById('out-matte').checked,
        matte_format: 'exr',
        comp_enabled: document.getElementById('out-comp').checked,
        comp_format: 'png',
        processed_enabled: document.getElementById('out-processed').checked,
        processed_format: 'exr',
    };
}

// --- WebSocket ---

function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(`${proto}//${location.host}/ws`);

    state.ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        switch (msg.type) {
            case 'progress':
                updateJobProgress(msg);
                break;
            case 'job_started':
            case 'job_completed':
            case 'job_failed':
            case 'job_cancelled':
                fetchJobs();
                break;
            case 'clips_updated':
                fetchClips();
                break;
            case 'warning':
                console.warn('Job warning:', msg.message);
                break;
            case 'status':
                updateJobStatus(msg);
                break;
        }
    };

    state.ws.onclose = () => {
        setTimeout(connectWS, 2000);
    };

    state.ws.onerror = () => {
        state.ws.close();
    };

    // Keep alive
    setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send('ping');
        }
    }, 30000);
}

function updateJobProgress(msg) {
    const jobEl = document.querySelector(`[data-job-id="${msg.job_id}"]`);
    if (!jobEl) return;

    const fill = jobEl.querySelector('.job-progress-fill');
    const text = jobEl.querySelector('.job-progress-text');
    if (fill && msg.total > 0) {
        const pct = (msg.current / msg.total) * 100;
        fill.style.width = pct + '%';
    }
    if (text) {
        text.textContent = `${msg.current}/${msg.total}`;
    }
}

function updateJobStatus(msg) {
    const jobEl = document.querySelector(`[data-job-id="${msg.job_id}"]`);
    if (!jobEl) return;
    const statusEl = jobEl.querySelector('.job-type');
    if (statusEl) {
        statusEl.textContent = msg.message;
    }
}

// --- Rendering ---

function renderClipList() {
    const container = document.getElementById('clip-list');
    const countEl = document.getElementById('clip-count');
    countEl.textContent = state.clips.length;

    if (state.clips.length === 0) {
        container.innerHTML = '<div class="empty-state">No clips loaded. Enter a path and click Scan.</div>';
        return;
    }

    container.innerHTML = state.clips.map((clip, i) => {
        const selected = state.selectedClip === clip.name ? 'selected' : '';
        const thumbUrl = `/api/clips/${encodeURIComponent(clip.name)}/thumbnail`;
        const actions = getClipActions(clip);

        return `
            <div class="clip-card ${selected}" onclick="selectClip('${escapeAttr(clip.name)}')"
                 style="animation-delay: ${i * 30}ms">
                <img class="clip-thumb" src="${thumbUrl}" alt=""
                     onerror="this.style.background='var(--surface-elevated)';this.alt='No preview'">
                <div class="clip-info">
                    <div class="clip-name" title="${escapeAttr(clip.name)}">${escapeHtml(clip.name)}</div>
                    <div class="clip-meta">
                        <span class="state-badge state-${clip.state}">${clip.state}</span>
                        <span>${clip.frame_count} frames</span>
                        ${clip.completed_frames > 0 ? `<span>${clip.completed_frames} done</span>` : ''}
                    </div>
                    ${actions ? `<div class="clip-actions">${actions}</div>` : ''}
                </div>
            </div>
        `;
    }).join('');
}

function getClipActions(clip) {
    const btns = [];
    const name = escapeAttr(clip.name);

    if (clip.state === 'COMPLETE') {
        btns.push(`<button class="btn btn-sm btn-accent" onclick="event.stopPropagation(); openOutputInFinder('${name}')" title="Show output in Finder">📂 Open</button>`);
        btns.push(`<button class="btn btn-sm btn-accent" onclick="event.stopPropagation(); copyOutputToSibling('${name}')" title="Copy FG output to ${escapeAttr(clip.name)}_KEYED folder">📋 Export</button>`);
    }
    if (clip.state === 'READY' || clip.state === 'COMPLETE') {
        btns.push(`<button class="btn btn-sm btn-ghost" onclick="event.stopPropagation(); submitInference('${name}')">Process</button>`);
    }
    if (clip.state === 'RAW') {
        btns.push(`<button class="btn btn-sm btn-accent" onclick="event.stopPropagation(); openAlphaImport('${name}')" title="Import alpha hint from a folder of images">Import Alpha</button>`);
        btns.push(`<button class="btn btn-sm btn-ghost" onclick="event.stopPropagation(); submitGVM('${name}')">GVM</button>`);
    }
    if (clip.state === 'MASKED' || clip.state === 'RAW') {
        btns.push(`<button class="btn btn-sm btn-ghost" onclick="event.stopPropagation(); submitVideoMaMa('${name}')">VMaMa</button>`);
    }
    if (clip.state === 'EXTRACTING') {
        btns.push(`<button class="btn btn-sm btn-accent" onclick="event.stopPropagation(); submitExtract('${name}')" title="Extract video frames">Extract</button>`);
    }
    if (clip.has_outputs) {
        btns.push(`<button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); deleteClipOutputs('${name}')" title="Clear outputs">Clear</button>`);
    }
    btns.push(`<button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); deleteClipEntirely('${name}')" title="Delete clip entirely">Del</button>`);

    return btns.join('');
}

function selectClip(name) {
    state.selectedClip = name;
    state.previewFrame = 0;
    renderClipList();
    renderPreview();
}

function setLayer(layer) {
    state.previewLayer = layer;
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.layer === layer);
    });
    renderPreview();
}

function renderPreview() {
    const container = document.getElementById('preview-container');
    const scrubber = document.getElementById('frame-scrubber');

    if (!state.selectedClip) {
        container.innerHTML = '<div class="empty-state">Select a clip to preview</div>';
        scrubber.style.display = 'none';
        return;
    }

    const clip = state.clips.find(c => c.name === state.selectedClip);
    if (!clip) {
        container.innerHTML = '<div class="empty-state">Clip not found</div>';
        scrubber.style.display = 'none';
        return;
    }

    const url = `/api/clips/${encodeURIComponent(clip.name)}/preview/${state.previewFrame}?layer=${state.previewLayer}&_t=${Date.now()}`;
    container.innerHTML = `<img src="${url}" alt="Preview" onerror="this.parentNode.innerHTML='<div class=\\'empty-state\\'>No preview for this layer/frame</div>'">`;

    // Frame scrubber
    const maxFrame = Math.max(0, clip.frame_count - 1);
    const slider = document.getElementById('frame-slider');
    slider.max = maxFrame;
    slider.value = state.previewFrame;
    document.getElementById('frame-label').textContent = `${state.previewFrame} / ${maxFrame}`;
    scrubber.style.display = maxFrame > 0 ? 'flex' : 'none';
}

function onFrameSlide(val) {
    state.previewFrame = parseInt(val, 10);
    document.getElementById('frame-label').textContent =
        `${state.previewFrame} / ${document.getElementById('frame-slider').max}`;
    renderPreview();
}

function renderJobQueue() {
    const container = document.getElementById('job-list');
    const countEl = document.getElementById('job-count');

    // Filter to active/recent jobs
    const activeJobs = state.jobs.filter(j => j.status === 'running' || j.status === 'queued');
    const recentDone = state.jobs.filter(j => j.status !== 'running' && j.status !== 'queued').slice(-5);
    const display = [...activeJobs, ...recentDone];

    countEl.textContent = activeJobs.length;

    if (display.length === 0) {
        container.innerHTML = '<div class="empty-state empty-state-small">No active jobs</div>';
        return;
    }

    let html = '';
    if (recentDone.length > 0) {
        html += `<div style="text-align:right;margin-bottom:6px"><button class="btn btn-sm btn-ghost" onclick="clearJobHistory()">Clear History</button></div>`;
    }

    html += display.map(job => {
        const pct = job.total_frames > 0 ? ((job.current_frame / job.total_frames) * 100).toFixed(0) : 0;
        const isActive = job.status === 'running' || job.status === 'queued';

        return `
            <div class="job-item" data-job-id="${job.id}">
                <div class="job-info">
                    <div class="job-title">${escapeHtml(job.clip_name)}</div>
                    <div class="job-type">${job.job_type.replace('_', ' ')}</div>
                </div>
                ${job.status === 'running' ? `
                    <div class="job-progress-bar">
                        <div class="job-progress-fill" style="width: ${pct}%"></div>
                    </div>
                    <span class="job-progress-text">${job.current_frame}/${job.total_frames}</span>
                ` : `
                    <span class="job-status ${job.status}">${job.status}</span>
                `}
                ${isActive ? `
                    <button class="btn btn-sm btn-danger" onclick="cancelJob('${job.id}')">Cancel</button>
                ` : ''}
                ${job.error_message ? `<span class="job-status failed" title="${escapeAttr(job.error_message)}">!</span>` : ''}
            </div>
        `;
    }).join('');

    container.innerHTML = html;
}

async function clearJobHistory() {
    try {
        await fetch('/api/jobs/clear', { method: 'POST' });
        fetchJobs();
    } catch (e) {
        console.error('Clear history failed:', e);
    }
}

// --- File Browser ---

let browserCurrentPath = '~';
let browserMode = 'scan'; // 'scan' or 'alpha'
let alphaTargetClip = null;

async function openBrowser() {
    browserMode = 'scan';
    alphaTargetClip = null;
    const startPath = document.getElementById('path-input').value.trim() || '~';
    document.getElementById('browser-modal').style.display = 'flex';
    document.getElementById('browser-title').textContent = 'Locate Clips Folder';
    document.getElementById('browser-select-btn').textContent = 'Select This Folder';
    document.getElementById('browser-select-btn').style.display = '';
    await browseTo(startPath);
}

async function openAlphaImport(clipName) {
    browserMode = 'alpha';
    alphaTargetClip = clipName;
    const startPath = document.getElementById('path-input').value.trim() || '~';
    document.getElementById('browser-modal').style.display = 'flex';
    document.getElementById('browser-title').textContent = 'Import Alpha Hint for: ' + clipName;
    document.getElementById('browser-select-btn').textContent = 'Use This Folder as Alpha';
    document.getElementById('browser-select-btn').style.display = '';
    await browseTo(startPath);
}

function closeBrowser() {
    document.getElementById('browser-modal').style.display = 'none';
}

async function browseTo(path) {
    try {
        const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Cannot browse this path', 'error');
            return;
        }
        const data = await res.json();
        browserCurrentPath = data.current;
        renderBreadcrumb(data.current);
        renderBrowserList(data);
        document.getElementById('browser-path').textContent = data.current;
    } catch (e) {
        showToast('Browse failed: ' + e.message, 'error');
    }
}

function renderBreadcrumb(path) {
    const el = document.getElementById('browser-breadcrumb');
    const parts = path.split('/').filter(Boolean);
    let html = `<button class="breadcrumb-part" onclick="browseTo('/')">/</button>`;
    let accumulated = '';
    for (const part of parts) {
        accumulated += '/' + part;
        const p = accumulated;
        html += `<span class="breadcrumb-sep">/</span>`;
        html += `<button class="breadcrumb-part" onclick="browseTo('${escapeAttr(p)}')">${escapeHtml(part)}</button>`;
    }
    el.innerHTML = html;
}

function renderBrowserList(data) {
    const el = document.getElementById('browser-list');
    const items = data.items;

    if (items.length === 0) {
        el.innerHTML = '<div class="browser-empty">Empty folder</div>';
        return;
    }

    // Directories first, then files
    const dirs = items.filter(i => i.type === 'dir');
    const files = items.filter(i => i.type === 'file');

    let html = '';

    // Parent directory link
    if (data.parent) {
        html += `
            <div class="browser-item" ondblclick="browseTo('${escapeAttr(data.parent)}')">
                <span class="browser-icon dir">..</span>
                <span class="browser-name">(parent)</span>
            </div>`;
    }

    for (const item of dirs) {
        html += `
            <div class="browser-item" ondblclick="browseTo('${escapeAttr(item.path)}')">
                <span class="browser-icon dir">&#128193;</span>
                <span class="browser-name">${escapeHtml(item.name)}</span>
            </div>`;
    }

    for (const item of files) {
        html += `
            <div class="browser-item browser-file-selectable" onclick="selectBrowserFile('${escapeAttr(item.path)}')">
                <span class="browser-icon file">&#127916;</span>
                <span class="browser-name">${escapeHtml(item.name)}</span>
                <span class="browser-file-add">+ Add</span>
            </div>`;
    }

    el.innerHTML = html;
}

async function selectBrowserPath() {
    if (browserMode === 'alpha' && alphaTargetClip) {
        closeBrowser();
        try {
            const res = await fetch('/api/clips/import-alpha', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clip_name: alphaTargetClip, alpha_path: browserCurrentPath }),
            });
            if (!res.ok) {
                const err = await res.json();
                showToast(err.detail || 'Failed to import alpha', 'error');
                return;
            }
            const data = await res.json();
            showToast(`Imported ${data.alpha_frames} alpha frame(s) for "${data.clip}"`, 'success', 5000);
            fetchClips();
        } catch (e) {
            showToast('Failed to import alpha: ' + e.message, 'error');
        }
        return;
    }
    document.getElementById('path-input').value = browserCurrentPath;
    closeBrowser();
    scanClips();
}

async function selectBrowserFile(filePath) {
    closeBrowser();
    if (browserMode === 'alpha' && alphaTargetClip) {
        try {
            const res = await fetch('/api/clips/import-alpha', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clip_name: alphaTargetClip, alpha_path: filePath }),
            });
            if (!res.ok) {
                const err = await res.json();
                showToast(err.detail || 'Failed to import alpha', 'error');
                return;
            }
            const data = await res.json();
            showToast(`Imported ${data.alpha_frames} alpha frame(s) for "${data.clip}"`, 'success', 5000);
            fetchClips();
        } catch (e) {
            showToast('Failed to import alpha: ' + e.message, 'error');
        }
        return;
    }
    try {
        const res = await fetch('/api/clips/add-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: filePath }),
        });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Failed to add file', 'error');
            return;
        }
        const data = await res.json();
        // Update the path input to show the parent folder
        document.getElementById('path-input').value = data.clips_dir;
        fetchClips();
        fetchJobs();
    } catch (e) {
        showToast('Failed to add file: ' + e.message, 'error');
    }
}

// --- Toast Notifications ---

function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = { success: '✓', error: '✗', info: 'ℹ', warning: '⚠' };
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;

    toast.onclick = () => dismissToast(toast);
    container.appendChild(toast);

    setTimeout(() => dismissToast(toast), duration);
    return toast;
}

function dismissToast(toast) {
    if (toast.classList.contains('toast-exit')) return;
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 200);
}

// --- Open Output / Copy Output ---

async function openOutputInFinder(clipName) {
    try {
        const res = await fetch('/api/clips/open-output', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: clipName }),
        });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Could not open folder', 'error');
            return;
        }
        showToast('Opened output folder in Finder', 'success');
    } catch (e) {
        showToast('Failed to open folder: ' + e.message, 'error');
    }
}

async function copyOutputToSibling(clipName) {
    try {
        const res = await fetch('/api/clips/copy-output', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: clipName }),
        });
        if (!res.ok) {
            const err = await res.json();
            showToast(err.detail || 'Copy failed', 'error');
            return;
        }
        const data = await res.json();
        showToast(`Copied ${data.files_copied} files to ${clipName}_KEYED`, 'success', 5000);
    } catch (e) {
        showToast('Copy failed: ' + e.message, 'error');
    }
}

// --- Utilities ---

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/'/g, '&#39;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}
