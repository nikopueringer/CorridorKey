<script lang="ts">
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import { api } from '$lib/api';
	import type { Clip, InferenceParams, OutputConfig } from '$lib/api';
	import { defaultParams, defaultOutputConfig } from '$lib/stores/settings';
	import { refreshJobs } from '$lib/stores/jobs';
	import { refreshClips } from '$lib/stores/clips';
	import { toast } from '$lib/stores/toasts';
	import FrameViewer from '../../../components/FrameViewer.svelte';
	import InferenceForm from '../../../components/InferenceForm.svelte';

	let clip = $state<Clip | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let submitting = $state(false);

	let params = $state<InferenceParams>({ ...$defaultParams });
	let outputConfig = $state<OutputConfig>({ ...$defaultOutputConfig });

	let clipName = $derived(decodeURIComponent(page.params.name));

	let availablePasses = $derived<string[]>(() => {
		const passes = ['input'];
		if (clip?.alpha_asset) passes.push('alpha');
		if (clip?.has_outputs) passes.push('fg', 'matte', 'comp', 'processed');
		return passes;
	});

	let canExtract = $derived(clip?.state === 'EXTRACTING');
	let canRunInference = $derived(clip?.state === 'READY' || clip?.state === 'COMPLETE');
	let canRunGVM = $derived(clip?.state === 'RAW');
	let canRunVideoMaMa = $derived(clip?.state === 'MASKED');

	async function loadClip() {
		loading = true;
		error = null;
		try {
			clip = await api.clips.get(clipName);
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	let canRunPipeline = $derived(
		clip?.state === 'EXTRACTING' || clip?.state === 'RAW' || clip?.state === 'READY'
	);

	async function runPipeline() {
		if (!clip) return;
		submitting = true;
		try {
			await api.jobs.submitPipeline([clip.name], 'gvm', params, outputConfig);
			toast.success('Pipeline started');
			await refreshJobs();
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		} finally {
			submitting = false;
		}
	}

	async function extractFrames() {
		if (!clip) return;
		submitting = true;
		try {
			await api.jobs.submitExtract([clip.name]);
			await refreshJobs();
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		} finally {
			submitting = false;
		}
	}

	async function runInference() {
		if (!clip) return;
		submitting = true;
		try {
			await api.jobs.submitInference([clip.name], params, outputConfig);
			await refreshJobs();
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		} finally {
			submitting = false;
		}
	}

	async function runGVM() {
		if (!clip) return;
		submitting = true;
		try {
			await api.jobs.submitGVM([clip.name]);
			await refreshJobs();
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		} finally {
			submitting = false;
		}
	}

	async function runVideoMaMa() {
		if (!clip) return;
		submitting = true;
		try {
			await api.jobs.submitVideoMaMa([clip.name]);
			await refreshJobs();
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		} finally {
			submitting = false;
		}
	}

	let uploadingAlpha = $state(false);
	let uploadingMask = $state(false);

	async function onMaskUpload(e: Event) {
		const input = e.target as HTMLInputElement;
		if (!input.files?.length || !clip) return;
		uploadingMask = true;
		try {
			await api.upload.mask(clip.name, input.files[0]);
			toast.success('VideoMaMa mask uploaded');
			await refreshClips();
			await loadClip();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			uploadingMask = false;
			input.value = '';
		}
	}

	async function onAlphaUpload(e: Event) {
		const input = e.target as HTMLInputElement;
		if (!input.files?.length || !clip) return;
		uploadingAlpha = true;
		try {
			await api.upload.alpha(clip.name, input.files[0]);
			await refreshClips();
			await loadClip();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			uploadingAlpha = false;
			input.value = '';
		}
	}

	let deleting = $state(false);

	async function deleteClip() {
		if (!clip || !confirm(`Delete clip "${clip.name}" and all its files? This cannot be undone.`)) return;
		deleting = true;
		try {
			await api.clips.delete(clip.name);
			await refreshClips();
			goto('/clips');
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		} finally {
			deleting = false;
		}
	}

	onMount(loadClip);
</script>

<svelte:head>
	<title>{clipName} — CorridorKey</title>
</svelte:head>

<div class="page">
	<div class="page-header">
		<a href="/clips" class="back-link">
			<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M9 3L5 7l4 4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>
			Clips
		</a>
		<h1 class="page-title">{clipName}</h1>
		{#if clip}
			<span class="state-pill mono" style="--pill-color: var(--state-{clip.state.toLowerCase()})">{clip.state}</span>
		{/if}
	</div>

	{#if loading}
		<div class="loading-state">Loading...</div>
	{:else if error}
		<div class="error-banner mono">{error}</div>
	{:else if clip}
		<div class="detail-layout">
			<div class="detail-main">
				<FrameViewer
					clipName={clip.name}
					frameCount={clip.frame_count}
					availablePasses={availablePasses()}
				/>

				<div class="clip-info">
					<div class="info-row">
						<span class="info-label mono">FRAMES</span>
						<span class="info-val">{clip.frame_count}</span>
					</div>
					{#if clip.completed_frames > 0}
						<div class="info-row">
							<span class="info-label mono">COMPLETED</span>
							<span class="info-val" style="color: var(--state-complete)">{clip.completed_frames}</span>
						</div>
					{/if}
					{#if clip.input_asset}
						<div class="info-row">
							<span class="info-label mono">INPUT TYPE</span>
							<span class="info-val">{clip.input_asset.asset_type}</span>
						</div>
					{/if}
					{#if clip.alpha_asset}
						<div class="info-row">
							<span class="info-label mono">ALPHA FRAMES</span>
							<span class="info-val">{clip.alpha_asset.frame_count}</span>
						</div>
					{/if}
					{#if clip.error_message}
						<div class="info-row">
							<span class="info-label mono">ERROR</span>
							<span class="info-val" style="color: var(--state-error)">{clip.error_message}</span>
						</div>
					{/if}
				</div>
			</div>

			<div class="detail-sidebar">
				<InferenceForm bind:params bind:outputConfig />

				<div class="action-buttons">
					{#if canRunPipeline}
						<button class="btn btn-hero" onclick={runPipeline} disabled={submitting}>
							Run Full Pipeline
						</button>
						<div class="divider-label mono">OR RUN INDIVIDUAL STEPS</div>
					{/if}

					{#if canExtract}
						<button class="btn btn-primary" onclick={extractFrames} disabled={submitting}>
							Extract Frames
						</button>
					{/if}
					{#if canRunInference}
						<button class="btn btn-primary" onclick={runInference} disabled={submitting}>
							Run Inference
						</button>
					{/if}
					{#if canRunGVM}
						<button class="btn btn-secondary" onclick={runGVM} disabled={submitting}>
							Run GVM Alpha
						</button>
					{/if}
					{#if canRunVideoMaMa}
						<button class="btn btn-secondary" onclick={runVideoMaMa} disabled={submitting}>
							Run VideoMaMa
						</button>
					{/if}
					{#if canRunGVM || canRunVideoMaMa || canExtract}
						<label class="btn btn-outline" class:disabled={uploadingAlpha}>
							<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 2v8M3 6l4-4 4 4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 11h10" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>
							{uploadingAlpha ? 'Uploading...' : 'Upload Alpha Hints (.zip)'}
							<input type="file" accept=".zip" hidden oninput={onAlphaUpload} disabled={uploadingAlpha} />
						</label>
						<label class="btn btn-outline" class:disabled={uploadingMask}>
							<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 2v8M3 6l4-4 4 4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 11h10" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>
							{uploadingMask ? 'Uploading...' : 'Upload VideoMaMa Mask (.zip)'}
							<input type="file" accept=".zip" hidden oninput={onMaskUpload} disabled={uploadingMask} />
						</label>
					{/if}
					{#if !canRunInference && !canRunGVM && !canRunVideoMaMa && !canExtract && !canRunPipeline}
						<p class="no-actions mono">Clip is complete.</p>
					{/if}
					<button class="btn btn-danger" onclick={deleteClip} disabled={deleting || submitting}>
						{deleting ? 'Deleting...' : 'Delete Clip'}
					</button>
				</div>
			</div>
		</div>
	{/if}
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
		gap: var(--sp-3);
	}

	.back-link {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 12px;
		color: var(--text-tertiary);
		transition: color 0.1s;
	}

	.back-link:hover {
		color: var(--text-primary);
	}

	.page-title {
		font-family: var(--font-sans);
		font-size: 20px;
		font-weight: 700;
		letter-spacing: -0.01em;
		flex: 1;
	}

	.state-pill {
		padding: 3px 10px;
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.06em;
		color: var(--pill-color, var(--text-tertiary));
		border: 1px solid var(--pill-color, var(--border));
		border-radius: 4px;
	}

	.loading-state {
		padding: var(--sp-8);
		text-align: center;
		color: var(--text-tertiary);
		font-size: 13px;
	}

	.error-banner {
		padding: var(--sp-3) var(--sp-4);
		background: rgba(255, 82, 82, 0.06);
		border: 1px solid rgba(255, 82, 82, 0.2);
		border-radius: var(--radius-md);
		font-size: 12px;
		color: var(--state-error);
	}

	.detail-layout {
		display: grid;
		grid-template-columns: 1fr 320px;
		gap: var(--sp-6);
	}

	.detail-main {
		display: flex;
		flex-direction: column;
		gap: var(--sp-4);
	}

	.clip-info {
		display: flex;
		flex-direction: column;
		gap: var(--sp-2);
		padding: var(--sp-4);
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: 8px;
	}

	.info-row {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}

	.info-label {
		font-size: 10px;
		color: var(--text-tertiary);
		letter-spacing: 0.06em;
	}

	.info-val {
		font-size: 13px;
		color: var(--text-primary);
	}

	.detail-sidebar {
		display: flex;
		flex-direction: column;
		gap: var(--sp-5);
		padding: var(--sp-5);
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: 8px;
		align-self: start;
	}

	.action-buttons {
		display: flex;
		flex-direction: column;
		gap: var(--sp-2);
		padding-top: var(--sp-3);
		border-top: 1px solid var(--border);
	}

	.btn {
		padding: 10px var(--sp-5);
		font-size: 13px;
		font-weight: 600;
		border-radius: var(--radius-md);
		border: none;
		cursor: pointer;
		transition: all 0.2s;
		text-align: center;
	}

	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.btn-primary {
		background: var(--accent);
		color: #000;
		box-shadow: 0 0 12px rgba(255, 242, 3, 0.15);
	}

	.btn-primary:hover:not(:disabled) {
		background: #fff;
		color: #000;
		box-shadow: 0 0 24px rgba(255, 242, 3, 0.3);
		transform: translateY(-1px);
	}

	.btn-secondary {
		background: var(--secondary-muted);
		color: var(--secondary);
		border: 1px solid var(--secondary);
	}

	.btn-secondary:hover:not(:disabled) {
		background: rgba(0, 154, 218, 0.2);
		color: var(--secondary-hover);
		border-color: var(--secondary-hover);
	}

	.btn-hero {
		background: var(--accent);
		color: #000;
		font-size: 14px;
		padding: 12px var(--sp-5);
		box-shadow: 0 0 16px rgba(255, 242, 3, 0.2);
	}

	.btn-hero:hover:not(:disabled) {
		background: #fff;
		box-shadow: 0 0 24px rgba(255, 242, 3, 0.35);
		transform: translateY(-1px);
	}

	.divider-label {
		text-align: center;
		font-size: 9px;
		letter-spacing: 0.1em;
		color: var(--text-tertiary);
		padding: var(--sp-2) 0;
		position: relative;
	}

	.divider-label::before,
	.divider-label::after {
		content: '';
		position: absolute;
		top: 50%;
		height: 1px;
		background: var(--border);
		width: 25%;
	}

	.divider-label::before { left: 0; }
	.divider-label::after { right: 0; }

	.btn-outline {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: var(--sp-2);
		background: transparent;
		color: var(--text-secondary);
		border: 1px dashed var(--border);
		cursor: pointer;
		font-size: 12px;
	}

	.btn-outline:hover:not(.disabled) {
		color: var(--accent);
		border-color: var(--accent);
		border-style: solid;
	}

	.btn-outline.disabled {
		opacity: 0.4;
		pointer-events: none;
	}

	.btn-danger {
		background: transparent;
		color: var(--text-tertiary);
		border: 1px solid var(--border);
		margin-top: var(--sp-2);
	}

	.btn-danger:hover:not(:disabled) {
		color: var(--state-error);
		border-color: var(--state-error);
		background: rgba(255, 82, 82, 0.08);
	}

	.no-actions {
		font-size: 11px;
		color: var(--text-tertiary);
		text-align: center;
		padding: var(--sp-2);
	}
</style>
