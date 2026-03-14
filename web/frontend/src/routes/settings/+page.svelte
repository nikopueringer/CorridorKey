<script lang="ts">
	import { onMount } from 'svelte';
	import { defaultParams, defaultOutputConfig, autoExtractFrames } from '$lib/stores/settings';
	import { api } from '$lib/api';
	import type { WeightInfo } from '$lib/api';
	import { toast } from '$lib/stores/toasts';
	import InferenceForm from '../../components/InferenceForm.svelte';

	let unloading = $state(false);
	let weights = $state<Record<string, WeightInfo>>({});
	let weightsLoading = $state(true);
	let vramLimit = $state(0);
	let vramLimitSaving = $state(false);

	const weightLabels: Record<string, { name: string; desc: string }> = {
		corridorkey: { name: 'CorridorKey v1.0', desc: 'Core neural keyer model (~300 MB)' },
		gvm: { name: 'GVM (Alpha Generator)', desc: 'Automatic alpha hint generation (~10 GB, needs ~80 GB VRAM)' },
		videomama: { name: 'VideoMaMa', desc: 'Mask-guided alpha generation (~5 GB)' },
	};

	async function loadWeights() {
		weightsLoading = true;
		try {
			weights = await api.system.weights();
		} catch {
			// ignore
		} finally {
			weightsLoading = false;
		}
	}

	async function downloadWeight(name: string) {
		try {
			await api.system.downloadWeights(name);
			toast.info(`Downloading ${name} weights...`);
			let polls = 0;
			const maxPolls = 400; // ~20 min at 3s intervals
			const poll = setInterval(async () => {
				polls++;
				await loadWeights();
				const w = weights[name];
				if (!w?.download || w.download.status !== 'downloading' || polls >= maxPolls) {
					clearInterval(poll);
					if (w?.installed) toast.success(`${name} weights installed`);
					else if (w?.download?.status === 'failed') toast.error(`${name} download failed: ${w.download.error}`);
				}
			}, 3000);
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		}
	}

	async function handleUnload() {
		unloading = true;
		try {
			await api.system.unload();
		} finally {
			unloading = false;
		}
	}

	async function loadVramLimit() {
		try {
			const res = await api.system.getVramLimit();
			vramLimit = res.vram_limit_gb;
		} catch { /* ignore */ }
	}

	async function saveVramLimit() {
		vramLimitSaving = true;
		try {
			await api.system.setVramLimit(vramLimit);
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		} finally {
			vramLimitSaving = false;
		}
	}

	onMount(() => { loadWeights(); loadVramLimit(); });
</script>

<svelte:head>
	<title>Settings — CorridorKey</title>
</svelte:head>

<div class="page">
	<div class="page-header">
		<h1 class="page-title">Settings</h1>
	</div>

	<div class="settings-layout">
		<section class="settings-card">
			<h2 class="card-title mono">MODEL WEIGHTS</h2>
			<p class="card-desc">Download model weights required for inference and alpha generation.</p>

			{#if weightsLoading}
				<p class="card-desc">Checking weights...</p>
			{:else}
				<div class="weights-list">
					{#each Object.entries(weights) as [key, w]}
						{@const label = weightLabels[key]}
						<div class="weight-row">
							<div class="weight-info">
								<div class="weight-name">
									<span class="weight-dot" class:installed={w.installed}></span>
									{label?.name ?? key}
								</div>
								<span class="weight-desc">{label?.desc ?? ''}</span>
								{#if w.installed && w.detail}
									<span class="weight-detail mono">{w.detail}</span>
								{/if}
								{#if w.download?.status === 'failed' && w.download.error}
									<span class="weight-error mono">{w.download.error}</span>
								{/if}
							</div>
							<div class="weight-action">
								{#if w.installed}
									<span class="weight-badge mono installed">INSTALLED</span>
								{:else if w.download?.status === 'downloading'}
									<span class="weight-badge mono downloading">DOWNLOADING...</span>
								{:else}
									<button class="btn-sm" onclick={() => downloadWeight(key)}>
										Download ({w.size_hint})
									</button>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</section>

		<section class="settings-card">
			<h2 class="card-title mono">UPLOAD BEHAVIOR</h2>
			<label class="toggle-field">
				<input type="checkbox" bind:checked={$autoExtractFrames} class="toggle" />
				<div class="toggle-label">
					<span>Auto-extract frames on video upload</span>
					<span class="toggle-hint">When enabled, uploading a video automatically queues frame extraction.</span>
				</div>
			</label>
		</section>

		<section class="settings-card">
			<h2 class="card-title mono">DEFAULT PARAMETERS</h2>
			<p class="card-desc">Pre-fill values for the inference form.</p>
			<InferenceForm bind:params={$defaultParams} bind:outputConfig={$defaultOutputConfig} />
		</section>

		<section class="settings-card">
			<h2 class="card-title mono">GPU MANAGEMENT</h2>

			<div class="setting-row">
				<div class="setting-info">
					<span class="setting-label">Min Free VRAM for Parallel Jobs</span>
					<span class="setting-hint">GPU jobs won't start if free VRAM is below this. Set to 0 to disable (single job at a time).</span>
				</div>
				<div class="vram-limit-control">
					<input
						type="range"
						min="0"
						max="32"
						step="1"
						bind:value={vramLimit}
						onchange={saveVramLimit}
						class="limit-slider"
					/>
					<span class="limit-val mono">{vramLimit === 0 ? 'OFF' : `${vramLimit} GB`}</span>
				</div>
			</div>

			<button class="btn btn-secondary" onclick={handleUnload} disabled={unloading}>
				{unloading ? 'Unloading...' : 'Unload All Models'}
			</button>
		</section>
	</div>
</div>

<style>
	.page {
		padding: var(--sp-5) var(--sp-6);
		display: flex;
		flex-direction: column;
		gap: var(--sp-4);
	}

	.page-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.page-title {
		font-family: var(--font-sans);
		font-size: 20px;
		font-weight: 700;
		letter-spacing: -0.01em;
	}

	.settings-layout {
		display: flex;
		flex-direction: column;
		gap: var(--sp-4);
		max-width: 580px;
	}

	.settings-card {
		display: flex;
		flex-direction: column;
		gap: var(--sp-4);
		padding: var(--sp-5);
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
	}

	.card-title {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.12em;
		color: var(--accent);
	}

	.card-desc {
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.4;
	}

	/* Weights */
	.weights-list {
		display: flex;
		flex-direction: column;
		gap: 1px;
		border-radius: var(--radius-md);
		overflow: hidden;
	}

	.weight-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--sp-3);
		padding: var(--sp-3) var(--sp-4);
		background: var(--surface-3);
	}

	.weight-info {
		display: flex;
		flex-direction: column;
		gap: 2px;
		min-width: 0;
	}

	.weight-name {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
		font-size: 14px;
		font-weight: 600;
	}

	.weight-dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--text-tertiary);
		flex-shrink: 0;
	}

	.weight-dot.installed {
		background: var(--state-complete);
		box-shadow: 0 0 6px rgba(93, 216, 121, 0.3);
	}

	.weight-desc {
		font-size: 12px;
		color: var(--text-secondary);
		padding-left: 15px;
	}

	.weight-detail {
		font-size: 10px;
		color: var(--text-tertiary);
		padding-left: 15px;
	}

	.weight-error {
		font-size: 10px;
		color: var(--state-error);
		padding-left: 15px;
	}

	.weight-action {
		flex-shrink: 0;
	}

	.weight-badge {
		font-size: 9px;
		font-weight: 600;
		letter-spacing: 0.08em;
		padding: 3px 8px;
		border-radius: var(--radius-sm);
	}

	.weight-badge.installed {
		color: var(--state-complete);
		border: 1px solid var(--state-complete);
	}

	.weight-badge.downloading {
		color: var(--accent);
		border: 1px solid var(--accent);
		animation: pulse 1.5s ease-in-out infinite;
	}

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.5; }
	}

	.btn-sm {
		padding: 5px 12px;
		font-size: 12px;
		font-weight: 600;
		background: var(--accent);
		color: #000;
		border: none;
		border-radius: var(--radius-md);
		cursor: pointer;
		transition: all 0.15s;
		white-space: nowrap;
	}

	.btn-sm:hover {
		background: #fff;
		box-shadow: 0 0 12px rgba(255, 242, 3, 0.2);
	}

	/* Toggle */
	.toggle-field {
		display: flex;
		align-items: flex-start;
		gap: var(--sp-3);
		cursor: pointer;
	}

	.toggle {
		-webkit-appearance: none;
		appearance: none;
		width: 32px;
		height: 16px;
		border-radius: 8px;
		background: var(--surface-4);
		position: relative;
		cursor: pointer;
		transition: background 0.15s;
		flex-shrink: 0;
		margin-top: 2px;
	}

	.toggle::after {
		content: '';
		position: absolute;
		top: 2px;
		left: 2px;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--text-tertiary);
		transition: transform 0.15s, background 0.15s;
	}

	.toggle:checked {
		background: var(--accent-muted);
	}

	.toggle:checked::after {
		transform: translateX(16px);
		background: var(--accent);
	}

	.toggle-label {
		display: flex;
		flex-direction: column;
		gap: 2px;
		font-size: 13px;
		color: var(--text-primary);
	}

	.toggle-hint {
		font-size: 12px;
		color: var(--text-tertiary);
		line-height: 1.4;
	}

	/* VRAM limit */
	.setting-row {
		display: flex;
		flex-direction: column;
		gap: var(--sp-2);
	}

	.setting-info {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.setting-label {
		font-size: 13px;
		font-weight: 500;
		color: var(--text-primary);
	}

	.setting-hint {
		font-size: 12px;
		color: var(--text-tertiary);
		line-height: 1.4;
	}

	.vram-limit-control {
		display: flex;
		align-items: center;
		gap: var(--sp-3);
	}

	.limit-slider {
		flex: 1;
		-webkit-appearance: none;
		appearance: none;
		height: 4px;
		background: var(--surface-4);
		border-radius: 2px;
		outline: none;
		cursor: pointer;
	}

	.limit-slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: var(--accent);
		cursor: pointer;
		border: 2px solid var(--surface-2);
	}

	.limit-slider::-moz-range-thumb {
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: var(--accent);
		cursor: pointer;
		border: 2px solid var(--surface-2);
	}

	.limit-val {
		font-size: 12px;
		font-weight: 600;
		color: var(--accent);
		min-width: 45px;
		text-align: right;
	}

	/* Buttons */
	.btn {
		padding: 8px var(--sp-4);
		font-size: 13px;
		font-weight: 600;
		border-radius: var(--radius-md);
		border: none;
		cursor: pointer;
		transition: all 0.15s;
		align-self: flex-start;
	}

	.btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.btn-secondary {
		background: var(--surface-4);
		color: var(--text-primary);
		border: 1px solid var(--border);
	}

	.btn-secondary:hover:not(:disabled) {
		background: var(--surface-3);
		border-color: var(--text-tertiary);
	}
</style>
