<script lang="ts">
	import { api } from '$lib/api';
	import { onMount } from 'svelte';

	let {
		clipName,
		frameCount = 0,
		availablePasses = ['input'],
	}: {
		clipName: string;
		frameCount?: number;
		availablePasses?: string[];
	} = $props();

	let currentFrame = $state(0);
	let selectedPass = $state('input');
	let loading = $state(false);
	let error = $state(false);
	let mode = $state<'frame' | 'video' | 'compare'>('frame');
	let playbackFps = $state(24);
	let comparePass = $state('input');

	let compareUrl = $derived(
		frameCount > 0 ? api.preview.url(clipName, comparePass, currentFrame) : null
	);

	let imgUrl = $derived(
		frameCount > 0 ? api.preview.url(clipName, selectedPass, currentFrame) : null
	);

	let videoUrl = $derived(
		frameCount > 0 ? `/api/preview/${encodeURIComponent(clipName)}/${selectedPass}/video?fps=${playbackFps}` : null
	);

	let downloadUrl = $derived(
		frameCount > 0 ? `/api/preview/${encodeURIComponent(clipName)}/${selectedPass}/download` : null
	);

	const passLabels: Record<string, string> = {
		input: 'Input',
		alpha: 'Alpha Hint',
		fg: 'FG',
		matte: 'Matte',
		comp: 'Comp',
		processed: 'Processed',
	};

	function onImgLoad() { loading = false; error = false; }
	function onImgError() { loading = false; error = true; }

	function onFrameChange(e: Event) {
		const target = e.target as HTMLInputElement;
		currentFrame = parseInt(target.value, 10);
		loading = true;
		error = false;
	}

	function switchPass(pass: string) {
		selectedPass = pass;
		if (mode === 'frame') {
			loading = true;
			error = false;
		}
	}

	function onKeydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
		if (mode === 'video') return; // let video handle its own controls
		switch (e.key) {
			case 'ArrowLeft': e.preventDefault(); currentFrame = Math.max(0, currentFrame - 1); break;
			case 'ArrowRight': e.preventDefault(); currentFrame = Math.min(frameCount - 1, currentFrame + 1); break;
			case 'Home': e.preventDefault(); currentFrame = 0; break;
			case 'End': e.preventDefault(); currentFrame = frameCount - 1; break;
		}
	}
</script>

<svelte:window onkeydown={onKeydown} />

<div class="viewer">
	<div class="viewer-viewport" class:split={mode === 'compare'}>
		{#if mode === 'compare'}
			<div class="compare-side">
				{#if compareUrl}
					<img src={compareUrl} alt="Compare — {comparePass}" />
				{/if}
				<span class="compare-label mono">{passLabels[comparePass] ?? comparePass}</span>
			</div>
			<div class="compare-divider"></div>
			<div class="compare-side">
				{#if imgUrl}
					<img src={imgUrl} alt="Frame {currentFrame} — {selectedPass}" />
				{/if}
				<span class="compare-label mono">{passLabels[selectedPass] ?? selectedPass}</span>
			</div>
		{:else if mode === 'video' && videoUrl}
			<!-- svelte-ignore a11y_media_has_caption -->
			<video
				src={videoUrl}
				controls
				loop
				autoplay
				class="video-player"
			></video>
		{:else if imgUrl && !error}
			<img
				src={imgUrl}
				alt="Frame {currentFrame} — {selectedPass}"
				class:loading
				onload={onImgLoad}
				onerror={onImgError}
			/>
		{/if}
		{#if loading && mode === 'frame'}
			<div class="overlay"><div class="spinner"></div></div>
		{/if}
		{#if error && mode === 'frame'}
			<div class="overlay"><span class="mono">Frame unavailable</span></div>
		{/if}
		{#if !imgUrl && !videoUrl}
			<div class="overlay"><span class="mono">No frames</span></div>
		{/if}
		{#if mode !== 'video' && frameCount > 0}
			<div class="frame-counter mono">{currentFrame + 1} / {frameCount}</div>
		{/if}
	</div>

	<div class="viewer-controls">
		<div class="controls-row">
			<div class="pass-tabs">
				{#each availablePasses as pass}
					<button
						class="pass-tab mono"
						class:active={selectedPass === pass}
						onclick={() => switchPass(pass)}
					>
						{passLabels[pass] ?? pass}
					</button>
				{/each}
			</div>

			<div class="mode-actions">
				{#if frameCount > 1}
					<div class="mode-toggle">
						<button class="mode-btn mono" class:active={mode === 'frame'} onclick={() => { mode = 'frame'; }}>
							Frames
						</button>
						<button class="mode-btn mono" class:active={mode === 'video'} onclick={() => { mode = 'video'; }}>
							Play
						</button>
						<button class="mode-btn mono" class:active={mode === 'compare'} onclick={() => { mode = 'compare'; }}>
							A/B
						</button>
					</div>
				{/if}
				{#if downloadUrl}
					<a href={downloadUrl} class="dl-btn mono" title="Download {passLabels[selectedPass] ?? selectedPass} as ZIP">
						<svg width="12" height="12" viewBox="0 0 14 14" fill="none">
							<path d="M7 2v7M3 9l4 3 4-3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
							<path d="M2 12h10" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
						</svg>
						{passLabels[selectedPass] ?? selectedPass}
					</a>
				{/if}
			</div>
		</div>

		{#if (mode === 'frame' || mode === 'compare') && frameCount > 1}
			<div class="scrub-row">
				<button class="tbtn" onclick={() => { currentFrame = Math.max(0, currentFrame - 1); }} title="Previous frame">
					<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M4 6l4-3v6z" fill="currentColor"/></svg>
				</button>
				<input
					type="range"
					min="0"
					max={frameCount - 1}
					value={currentFrame}
					oninput={onFrameChange}
					class="scrub-slider"
				/>
				<button class="tbtn" onclick={() => { currentFrame = Math.min(frameCount - 1, currentFrame + 1); }} title="Next frame">
					<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M8 6l-4-3v6z" fill="currentColor"/></svg>
				</button>
			</div>
		{/if}

		{#if mode === 'compare'}
			<div class="compare-controls">
				<span class="compare-ctrl-label mono">COMPARE LEFT</span>
				<div class="pass-tabs">
					{#each availablePasses as pass}
						<button
							class="pass-tab mono"
							class:active={comparePass === pass}
							onclick={() => { comparePass = pass; }}
						>
							{passLabels[pass] ?? pass}
						</button>
					{/each}
				</div>
			</div>
		{/if}

		{#if mode === 'video' && frameCount > 1}
			<div class="fps-row">
				<span class="fps-label mono">FPS</span>
				<select bind:value={playbackFps} class="fps-select mono">
					<option value={8}>8</option>
					<option value={12}>12</option>
					<option value={24}>24</option>
					<option value={30}>30</option>
				</select>
				<span class="fps-hint mono">Change FPS to re-encode preview</span>
			</div>
		{/if}
	</div>
</div>

<style>
	.viewer {
		display: flex;
		flex-direction: column;
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		overflow: hidden;
		background: var(--surface-1);
		box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
	}

	.viewer-viewport {
		position: relative;
		aspect-ratio: 16 / 9;
		background:
			repeating-conic-gradient(var(--surface-3) 0% 25%, var(--surface-2) 0% 50%) 0 0 / 16px 16px;
		overflow: hidden;
	}

	.viewer-viewport img {
		width: 100%;
		height: 100%;
		object-fit: contain;
		transition: opacity 0.1s;
	}

	.viewer-viewport img.loading {
		opacity: 0.4;
	}

	.video-player {
		width: 100%;
		height: 100%;
		object-fit: contain;
		background: #000;
	}

	.viewer-viewport.split {
		display: flex;
	}

	.compare-side {
		flex: 1;
		position: relative;
		overflow: hidden;
	}

	.compare-side img {
		width: 100%;
		height: 100%;
		object-fit: contain;
	}

	.compare-label {
		position: absolute;
		top: 8px;
		left: 8px;
		padding: 2px 8px;
		font-size: 10px;
		font-weight: 600;
		color: var(--text-primary);
		background: rgba(0, 0, 0, 0.75);
		border-radius: var(--radius-sm);
		letter-spacing: 0.04em;
	}

	.compare-divider {
		width: 2px;
		background: var(--accent);
		flex-shrink: 0;
		box-shadow: 0 0 8px rgba(255, 242, 3, 0.3);
	}

	.compare-controls {
		display: flex;
		align-items: center;
		gap: var(--sp-3);
	}

	.compare-ctrl-label {
		font-size: 9px;
		color: var(--text-tertiary);
		letter-spacing: 0.08em;
		flex-shrink: 0;
	}

	.overlay {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--text-tertiary);
		font-size: 13px;
	}

	.spinner {
		width: 20px;
		height: 20px;
		border: 2px solid var(--surface-4);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	.frame-counter {
		position: absolute;
		bottom: var(--sp-2);
		right: var(--sp-2);
		padding: 2px 6px;
		font-size: 10px;
		color: var(--text-primary);
		background: rgba(0, 0, 0, 0.7);
		border-radius: var(--radius-sm);
		pointer-events: none;
	}

	@keyframes spin { to { transform: rotate(360deg); } }

	.viewer-controls {
		padding: var(--sp-3);
		display: flex;
		flex-direction: column;
		gap: var(--sp-3);
		border-top: 1px solid var(--border);
		background: var(--surface-2);
	}

	.controls-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--sp-3);
	}

	.pass-tabs {
		display: flex;
		gap: 2px;
		flex-wrap: wrap;
	}

	.pass-tab {
		padding: 3px 8px;
		font-size: 10px;
		font-weight: 500;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--surface-3);
		color: var(--text-secondary);
		cursor: pointer;
		transition: all 0.1s;
	}

	.pass-tab:hover { color: var(--text-primary); border-color: var(--text-tertiary); }
	.pass-tab.active { color: var(--accent); border-color: var(--accent); background: var(--accent-muted); }

	.mode-actions {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
		flex-shrink: 0;
	}

	.mode-toggle {
		display: flex;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		overflow: hidden;
	}

	.mode-btn {
		padding: 3px 10px;
		font-size: 10px;
		font-weight: 500;
		border: none;
		background: var(--surface-3);
		color: var(--text-secondary);
		cursor: pointer;
		transition: all 0.1s;
	}

	.mode-btn:first-child { border-right: 1px solid var(--border); }
	.mode-btn:hover { color: var(--text-primary); }
	.mode-btn.active { color: var(--accent); background: var(--accent-muted); }

	.dl-btn {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 3px 8px;
		height: 24px;
		font-size: 10px;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-secondary);
		transition: all 0.1s;
	}

	.dl-btn:hover { color: var(--accent); border-color: var(--accent); }

	.scrub-row {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
	}

	.tbtn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 24px;
		height: 22px;
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		background: var(--surface-3);
		color: var(--text-secondary);
		cursor: pointer;
		transition: all 0.1s;
		flex-shrink: 0;
	}

	.tbtn:hover { color: var(--text-primary); background: var(--surface-4); }

	.scrub-slider {
		flex: 1;
		-webkit-appearance: none;
		appearance: none;
		height: 4px;
		background: var(--surface-4);
		border-radius: 2px;
		outline: none;
		cursor: pointer;
	}

	.scrub-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--accent);
		cursor: pointer;
		border: 2px solid var(--surface-2);
	}

	.scrub-slider::-moz-range-thumb {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--accent);
		cursor: pointer;
		border: 2px solid var(--surface-2);
	}

	.fps-row {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
	}

	.fps-label {
		font-size: 10px;
		color: var(--text-tertiary);
	}

	.fps-select {
		padding: 2px 6px;
		font-size: 10px;
		background: var(--surface-3);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-secondary);
		cursor: pointer;
	}

	.fps-hint {
		font-size: 9px;
		color: var(--text-tertiary);
		margin-left: auto;
	}
</style>
