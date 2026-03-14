<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { refreshClips } from '$lib/stores/clips';
	import { refreshJobs } from '$lib/stores/jobs';
	import { autoExtractFrames } from '$lib/stores/settings';
	import { api } from '$lib/api';
	import type { Project, Clip } from '$lib/api';
	import ClipCard from '../../components/ClipCard.svelte';
	import ContextMenu from '../../components/ContextMenu.svelte';
	import type { MenuItem } from '../../components/ContextMenu.svelte';
	import { toast } from '$lib/stores/toasts';

	let projects = $state<Project[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let uploading = $state(false);
	let uploadError = $state<string | null>(null);
	let dragOver = $state(false);
	let creatingProject = $state(false);
	let newProjectName = $state('');
	let showCreateForm = $state(false);

	// Context menu state
	let ctxVisible = $state(false);
	let ctxX = $state(0);
	let ctxY = $state(0);
	let ctxItems = $state<MenuItem[]>([]);

	function showProjectContext(e: MouseEvent, project: Project) {
		e.preventDefault();
		ctxX = e.clientX;
		ctxY = e.clientY;
		const clipNames = project.clips.map(c => c.name);
		ctxItems = [
			{
				label: `Process All (${project.clip_count} clips)`,
				disabled: project.clip_count === 0,
				action: async () => {
					try {
						await api.jobs.submitPipeline(clipNames);
						toast.success(`Pipeline started for ${clipNames.length} clips`);
						await refreshJobs();
					} catch (e) {
						toast.error(e instanceof Error ? e.message : String(e));
					}
				},
			},
			{ label: '---', action: () => {} },
			{
				label: 'Rename',
				action: async () => {
					const name = prompt('New project name:', project.display_name);
					if (name && name.trim()) {
						await api.projects.rename(project.name, name.trim());
						await loadProjects();
					}
				},
			},
			{ label: '---', action: () => {} },
			{
				label: 'Delete Project',
				danger: true,
				action: () => deleteProject(project.name, project.display_name),
			},
		];
		ctxVisible = true;
	}

	function showClipContext(e: MouseEvent, clip: Clip, project: Project) {
		e.preventDefault();
		const otherProjects = projects.filter(p => p.name !== project.name);
		const moveItems: MenuItem[] = otherProjects.map(p => ({
			label: p.display_name,
			action: async () => {
				await api.clips.move(clip.name, p.name);
				await loadProjects();
			},
		}));

		ctxItems = [
			{
				label: 'Open',
				action: () => goto(`/clips/${encodeURIComponent(clip.name)}`),
			},
			{
				label: 'Run Full Pipeline',
				disabled: clip.state === 'COMPLETE',
				action: async () => {
					await api.jobs.submitPipeline([clip.name]);
					await refreshJobs();
				},
			},
			{ label: '---', action: () => {} },
			...(moveItems.length > 0
				? [{ label: 'Move to...', disabled: true, action: () => {} }, ...moveItems, { label: '---', action: () => {} }]
				: []),
			{
				label: 'Delete Clip',
				danger: true,
				action: async () => {
					if (confirm(`Delete clip "${clip.name}"?`)) {
						await api.clips.delete(clip.name);
						await loadProjects();
					}
				},
			},
		];
		ctxX = e.clientX;
		ctxY = e.clientY;
		ctxVisible = true;
	}
	let collapsedProjects = $state<Set<string>>(new Set());

	const VIDEO_EXTS = ['.mp4', '.mov', '.avi', '.mkv', '.mxf', '.webm', '.m4v'];
	const isVideo = (name: string) => VIDEO_EXTS.some(ext => name.toLowerCase().endsWith(ext));
	const isZip = (name: string) => name.toLowerCase().endsWith('.zip');

	async function loadProjects() {
		loading = true;
		error = null;
		try {
			projects = await api.projects.list();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	async function handleFiles(files: FileList | File[]) {
		uploading = true;
		uploadError = null;
		try {
			for (const file of files) {
				if (isVideo(file.name)) {
					await api.upload.video(file, undefined, $autoExtractFrames);
				} else if (isZip(file.name)) {
					await api.upload.frames(file);
				} else {
					uploadError = `Unsupported: ${file.name}. Use videos (.mp4, .mov, etc.) or zipped frames (.zip).`;
					continue;
				}
			}
			await Promise.all([loadProjects(), refreshClips(), refreshJobs()]);
		} catch (e) {
			uploadError = e instanceof Error ? e.message : String(e);
		} finally {
			uploading = false;
		}
	}

	async function createProject() {
		if (!newProjectName.trim()) return;
		creatingProject = true;
		try {
			await api.projects.create(newProjectName.trim());
			newProjectName = '';
			showCreateForm = false;
			await loadProjects();
		} catch (e) {
			uploadError = e instanceof Error ? e.message : String(e);
		} finally {
			creatingProject = false;
		}
	}

	async function deleteProject(name: string, displayName: string) {
		if (!confirm(`Delete project "${displayName}" and ALL clips inside it? This cannot be undone.`)) return;
		try {
			await api.projects.delete(name);
			await Promise.all([loadProjects(), refreshClips()]);
		} catch (e) {
			toast.error(e instanceof Error ? e.message : String(e));
		}
	}

	function toggleProject(name: string) {
		const next = new Set(collapsedProjects);
		if (next.has(name)) next.delete(name);
		else next.add(name);
		collapsedProjects = next;
	}

	function onDrop(e: DragEvent) { e.preventDefault(); dragOver = false; if (e.dataTransfer?.files.length) handleFiles(e.dataTransfer.files); }
	function onDragOver(e: DragEvent) { e.preventDefault(); dragOver = true; }
	function onDragLeave() { dragOver = false; }
	function onFileInput(e: Event) { const input = e.target as HTMLInputElement; if (input.files?.length) { handleFiles(input.files); input.value = ''; } }
	function onCreateKeydown(e: KeyboardEvent) { if (e.key === 'Enter') createProject(); if (e.key === 'Escape') showCreateForm = false; }

	let totalClips = $derived(projects.reduce((sum, p) => sum + p.clips.length, 0));

	onMount(loadProjects);
</script>

<svelte:head>
	<title>Clips — CorridorKey</title>
</svelte:head>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="page" class:drag-over={dragOver} ondrop={onDrop} ondragover={onDragOver} ondragleave={onDragLeave}>
	<div class="page-header">
		<div class="header-left">
			<h1 class="page-title">Projects</h1>
			{#if !loading}
				<span class="header-count mono">{projects.length} projects &middot; {totalClips} clips</span>
			{/if}
		</div>
		<div class="header-actions">
			<button class="btn-ghost" onclick={() => { showCreateForm = !showCreateForm; }} title="New project">
				<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 2v10M2 7h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
				New Project
			</button>
			<label class="btn-accent" class:disabled={uploading}>
				<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M7 2v8M3 6l4-4 4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 11h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
				{uploading ? 'Uploading...' : 'Upload'}
				<input type="file" accept=".mp4,.mov,.avi,.mkv,.mxf,.webm,.m4v,.zip" multiple hidden oninput={onFileInput} disabled={uploading} />
			</label>
			<button class="btn-ghost" onclick={loadProjects} disabled={loading}>
				<svg width="14" height="14" viewBox="0 0 14 14" fill="none" class:spinning={loading}><path d="M12 7a5 5 0 11-1.5-3.5M12 2v3h-3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>
			</button>
		</div>
	</div>

	{#if showCreateForm}
		<div class="create-form">
			<input
				type="text"
				class="create-input"
				placeholder="Project name..."
				bind:value={newProjectName}
				onkeydown={onCreateKeydown}
				disabled={creatingProject}
			/>
			<button class="btn-sm" onclick={createProject} disabled={creatingProject || !newProjectName.trim()}>
				{creatingProject ? 'Creating...' : 'Create'}
			</button>
			<button class="btn-ghost-sm" onclick={() => { showCreateForm = false; }}>Cancel</button>
		</div>
	{/if}

	{#if error || uploadError}
		<div class="error-banner mono">
			{error || uploadError}
		</div>
	{/if}

	{#if dragOver}
		<div class="drop-overlay">
			<div class="drop-content">
				<svg width="36" height="36" viewBox="0 0 36 36" fill="none"><path d="M18 6v18M10 14l8-8 8 8" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 28h26" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round"/></svg>
				<span class="drop-text">Drop videos or zipped frames</span>
			</div>
		</div>
	{/if}

	{#if projects.length === 0 && !loading}
		<div class="empty-state">
			<svg width="48" height="48" viewBox="0 0 48 48" fill="none">
				<rect x="6" y="10" width="36" height="28" rx="4" stroke="var(--text-tertiary)" stroke-width="1.5"/>
				<path d="M14 10v28M34 10v28M6 19h8M34 19h8M6 29h8M34 29h8" stroke="var(--text-tertiary)" stroke-width="1.2"/>
			</svg>
			<p class="empty-text">No projects yet</p>
			<p class="empty-hint">Drag & drop video files here, or click Upload to get started.</p>
		</div>
	{:else}
		<div class="project-list">
			{#each projects as project (project.name)}
				{@const collapsed = collapsedProjects.has(project.name)}
				<div class="project-group">
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div class="project-header" onclick={() => toggleProject(project.name)} oncontextmenu={(e) => showProjectContext(e, project)}>
						<svg class="chevron" class:collapsed width="12" height="12" viewBox="0 0 12 12" fill="none">
							<path d="M3 4.5l3 3 3-3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
						</svg>
						<span class="project-name">{project.display_name}</span>
						<span class="project-count mono">{project.clip_count} clip{project.clip_count !== 1 ? 's' : ''}</span>
						{#if project.created}
							<span class="project-date mono">{new Date(project.created).toLocaleDateString()}</span>
						{/if}
						<button
							class="project-delete"
							title="Delete project"
							onclick={(e) => { e.stopPropagation(); deleteProject(project.name, project.display_name); }}
						>
							<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>
						</button>
					</div>
					{#if !collapsed}
						<div class="project-clips"
							ondragover={(e) => { e.preventDefault(); e.currentTarget.classList.add('drop-target'); }}
							ondragleave={(e) => { e.currentTarget.classList.remove('drop-target'); }}
							ondrop={async (e) => {
								e.preventDefault();
								e.currentTarget.classList.remove('drop-target');
								const clipName = e.dataTransfer?.getData('text/clip-name');
								if (clipName) {
									try {
										await api.clips.move(clipName, project.name);
										await loadProjects();
									} catch (err) {
										toast.error(err instanceof Error ? err.message : String(err));
									}
								}
							}}
						>
							{#if project.clips.length === 0}
								<p class="no-clips mono">Drop clips here or upload new ones</p>
							{:else}
								<div class="clip-grid">
									{#each project.clips as clip (clip.name)}
										<div class="clip-wrap"
											draggable="true"
											ondragstart={(e) => { e.dataTransfer?.setData('text/clip-name', clip.name); }}
											oncontextmenu={(e) => showClipContext(e, clip, project)}
										>
											<ClipCard {clip} />
											{#if projects.length > 1}
												<select
													class="move-select mono"
													value=""
													onchange={async (e) => {
														const target = (e.target as HTMLSelectElement).value;
														if (!target) return;
														try {
															await api.clips.move(clip.name, target);
															await loadProjects();
														} catch (err) {
															toast.error(err instanceof Error ? err.message : String(err));
														}
													}}
												>
													<option value="" disabled selected>Move to...</option>
													{#each projects.filter(p => p.name !== project.name) as other}
														<option value={other.name}>{other.display_name}</option>
													{/each}
												</select>
											{/if}
										</div>
									{/each}
								</div>
							{/if}
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>

<ContextMenu bind:visible={ctxVisible} x={ctxX} y={ctxY} items={ctxItems} />

<style>
	.page {
		padding: var(--sp-5) var(--sp-6);
		display: flex;
		flex-direction: column;
		gap: var(--sp-4);
		min-height: 100%;
		position: relative;
	}

	.page.drag-over { background: var(--accent-glow); }

	.page-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.header-left {
		display: flex;
		align-items: baseline;
		gap: var(--sp-3);
	}

	.page-title {
		font-family: var(--font-sans);
		font-size: 20px;
		font-weight: 700;
		letter-spacing: -0.01em;
	}

	.header-count {
		font-size: 12px;
		color: var(--text-tertiary);
	}

	.header-actions {
		display: flex;
		gap: var(--sp-2);
		align-items: center;
	}

	.btn-accent {
		display: inline-flex;
		align-items: center;
		gap: var(--sp-2);
		padding: 6px var(--sp-3);
		font-size: 13px;
		font-weight: 600;
		color: #000;
		background: var(--accent);
		border: none;
		border-radius: var(--radius-md);
		cursor: pointer;
		transition: all 0.15s;
	}
	.btn-accent:hover { background: #fff; box-shadow: 0 0 12px rgba(255, 242, 3, 0.25); }
	.btn-accent.disabled { opacity: 0.5; pointer-events: none; }

	.btn-ghost {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
		padding: 6px var(--sp-3);
		font-size: 13px;
		font-weight: 500;
		color: var(--text-secondary);
		background: transparent;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		cursor: pointer;
		transition: all 0.15s;
	}
	.btn-ghost:hover { color: var(--text-primary); border-color: var(--text-tertiary); background: var(--surface-2); }
	.btn-ghost:disabled { opacity: 0.5; cursor: not-allowed; }

	.spinning { animation: spin 1s linear infinite; }
	@keyframes spin { to { transform: rotate(360deg); } }

	/* Create project form */
	.create-form {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
		padding: var(--sp-3) var(--sp-4);
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
	}

	.create-input {
		flex: 1;
		padding: 6px var(--sp-3);
		font-size: 14px;
		background: var(--surface-3);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-primary);
		outline: none;
		font-family: inherit;
	}
	.create-input:focus { border-color: var(--accent); }
	.create-input::placeholder { color: var(--text-tertiary); }

	.btn-sm {
		padding: 6px 14px;
		font-size: 12px;
		font-weight: 600;
		background: var(--accent);
		color: #000;
		border: none;
		border-radius: var(--radius-sm);
		cursor: pointer;
	}
	.btn-sm:hover { background: #fff; }
	.btn-sm:disabled { opacity: 0.4; cursor: not-allowed; }

	.btn-ghost-sm {
		padding: 6px 10px;
		font-size: 12px;
		color: var(--text-tertiary);
		background: none;
		border: none;
		cursor: pointer;
	}
	.btn-ghost-sm:hover { color: var(--text-primary); }

	.error-banner {
		padding: var(--sp-3) var(--sp-4);
		background: rgba(255, 82, 82, 0.06);
		border: 1px solid rgba(255, 82, 82, 0.2);
		border-radius: var(--radius-md);
		font-size: 12px;
		color: var(--state-error);
	}

	.drop-overlay {
		position: absolute;
		inset: 0;
		z-index: 10;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(0, 0, 0, 0.85);
		border: 2px dashed var(--accent);
		border-radius: var(--radius-lg);
		pointer-events: none;
	}

	.drop-content {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: var(--sp-3);
	}

	.drop-text {
		font-size: 16px;
		font-weight: 600;
		color: var(--accent);
	}

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--sp-3);
		padding: var(--sp-10) 0;
		text-align: center;
	}
	.empty-text { font-size: 16px; font-weight: 500; color: var(--text-secondary); }
	.empty-hint { font-size: 13px; color: var(--text-tertiary); max-width: 300px; }

	/* Project groups */
	.project-list {
		display: flex;
		flex-direction: column;
		gap: var(--sp-3);
	}

	.project-group {
		border: 1px solid var(--border);
		border-radius: var(--radius-lg);
		overflow: hidden;
		background: var(--surface-1);
	}

	.project-header {
		display: flex;
		align-items: center;
		gap: var(--sp-3);
		width: 100%;
		padding: var(--sp-3) var(--sp-4);
		background: var(--surface-2);
		border: none;
		color: var(--text-primary);
		cursor: pointer;
		font-family: inherit;
		font-size: 14px;
		text-align: left;
		transition: background 0.1s;
	}

	.project-header:hover {
		background: var(--surface-3);
	}

	.chevron {
		transition: transform 0.15s;
		color: var(--text-tertiary);
		flex-shrink: 0;
	}

	.chevron.collapsed {
		transform: rotate(-90deg);
	}

	.project-name {
		font-weight: 600;
		flex: 1;
	}

	.project-count {
		font-size: 11px;
		color: var(--text-secondary);
	}

	.project-date {
		font-size: 10px;
		color: var(--text-tertiary);
	}

	.project-delete {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		border: none;
		border-radius: var(--radius-sm);
		background: transparent;
		color: var(--text-tertiary);
		cursor: pointer;
		transition: all 0.1s;
		flex-shrink: 0;
	}
	.project-delete:hover {
		color: var(--state-error);
		background: rgba(255, 82, 82, 0.1);
	}

	.project-clips {
		padding: var(--sp-3) var(--sp-4) var(--sp-4);
	}

	.no-clips {
		font-size: 12px;
		color: var(--text-tertiary);
		padding: var(--sp-2);
	}

	.project-clips :global(.drop-target) {
		background: var(--accent-glow);
		outline: 2px dashed var(--accent);
		outline-offset: -2px;
		border-radius: var(--radius-md);
	}

	.clip-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
		gap: var(--sp-3);
	}

	.clip-wrap {
		position: relative;
		cursor: grab;
	}

	.clip-wrap:active {
		cursor: grabbing;
		opacity: 0.7;
	}

	.move-select {
		position: absolute;
		bottom: var(--sp-2);
		left: var(--sp-2);
		right: var(--sp-2);
		padding: 4px 6px;
		font-size: 10px;
		background: rgba(0, 0, 0, 0.8);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-secondary);
		cursor: pointer;
		opacity: 0;
		transition: opacity 0.15s;
		backdrop-filter: blur(4px);
	}

	.clip-wrap:hover .move-select {
		opacity: 1;
	}
</style>
