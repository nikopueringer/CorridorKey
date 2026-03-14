<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import { onMount } from 'svelte';
	import { connect, disconnect, onMessage, isConnected } from '$lib/ws';
	import { refreshClips } from '$lib/stores/clips';
	import { refreshJobs, updateJobFromWS, currentJob, activeJobCount } from '$lib/stores/jobs';
	import { refreshDevice, refreshVRAM, device, vram, wsConnected } from '$lib/stores/system';
	import VramMeter from '../components/VramMeter.svelte';
	import ToastContainer from '../components/ToastContainer.svelte';
	import KeyboardHelp from '../components/KeyboardHelp.svelte';
	import { toast } from '$lib/stores/toasts';

	let { children } = $props();

	let connected = $state(false);

	const navItems = [
		{ href: '/clips', label: 'Clips', icon: 'film' },
		{ href: '/jobs', label: 'Jobs', icon: 'layers' },
		{ href: '/settings', label: 'Settings', icon: 'sliders' },
	];

	function isActive(href: string): boolean {
		return page.url.pathname === href || page.url.pathname.startsWith(href + '/');
	}

	onMount(() => {
		connect();
		refreshDevice();
		refreshVRAM();
		refreshClips();
		refreshJobs();

		const unsubWs = onMessage((msg) => {
			if (msg.type === 'job:progress') {
				const d = msg.data as { job_id: string; clip_name: string; current: number; total: number };
				const found = updateJobFromWS(d.job_id, {
					current_frame: d.current,
					total_frames: d.total,
					status: 'running',
				});
				// Job not in stores yet — fetch it
				if (!found) refreshJobs();
			} else if (msg.type === 'job:status') {
				const d = msg.data as { job_id: string; status: string; error?: string };
				updateJobFromWS(d.job_id, { status: d.status, error_message: d.error ?? null });
				if (d.status === 'completed') toast.success(`Job completed: ${d.job_id}`);
				else if (d.status === 'failed') toast.error(`Job failed: ${d.error ?? d.job_id}`);
				else if (d.status === 'cancelled') toast.warning(`Job cancelled: ${d.job_id}`);
				refreshJobs();
				refreshClips();
			} else if (msg.type === 'job:warning') {
				const d = msg.data as { message: string };
				toast.warning(d.message);
			} else if (msg.type === 'vram:update') {
				const d = msg.data as { total: number; allocated: number; free: number; reserved: number; name: string };
				vram.set({ ...d, available: true });
			} else if (msg.type === 'clip:state_changed') {
				refreshClips();
			}
		});

		const connectionCheck = setInterval(() => {
			connected = isConnected();
			wsConnected.set(connected);
		}, 1000);

		const vramInterval = setInterval(refreshVRAM, 10000);

		return () => {
			unsubWs();
			clearInterval(connectionCheck);
			clearInterval(vramInterval);
			disconnect();
		};
	});
</script>

<div class="shell">
	<nav class="sidebar">
		<div class="sidebar-top">
			<a href="/clips" class="logo">
				<img src="/Corridor_Digital_Logo.svg" alt="Corridor Digital" class="logo-img" />
				<span class="logo-product mono">CORRIDORKEY</span>
			</a>

			<div class="nav-links">
				{#each navItems as item}
					<a
						href={item.href}
						class="nav-link"
						class:active={isActive(item.href)}
					>
						<span class="nav-icon">
							{#if item.icon === 'film'}
								<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="2" y="3" width="12" height="10" rx="1.5" stroke="currentColor" stroke-width="1.2"/><path d="M5 3v10M11 3v10M2 6.5h3M11 6.5h3M2 9.5h3M11 9.5h3" stroke="currentColor" stroke-width="1.0"/></svg>
							{:else if item.icon === 'layers'}
								<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 8l6 3.5L14 8M2 10.5l6 3.5 6-3.5M2 5.5L8 9l6-3.5L8 2 2 5.5z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg>
							{:else if item.icon === 'sliders'}
								<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><line x1="2" y1="4" x2="14" y2="4" stroke="currentColor" stroke-width="1.2"/><line x1="2" y1="8" x2="14" y2="8" stroke="currentColor" stroke-width="1.2"/><line x1="2" y1="12" x2="14" y2="12" stroke="currentColor" stroke-width="1.2"/><circle cx="5" cy="4" r="1.5" fill="var(--surface-2)" stroke="currentColor" stroke-width="1.2"/><circle cx="10" cy="8" r="1.5" fill="var(--surface-2)" stroke="currentColor" stroke-width="1.2"/><circle cx="7" cy="12" r="1.5" fill="var(--surface-2)" stroke="currentColor" stroke-width="1.2"/></svg>
							{/if}
						</span>
						<span class="nav-label">{item.label}</span>
						{#if item.href === '/jobs' && $activeJobCount > 0}
							<span class="nav-badge mono">{$activeJobCount}</span>
						{/if}
					</a>
				{/each}
			</div>
		</div>

		<div class="sidebar-bottom">
			<VramMeter />
			<div class="device-row">
				<span class="device-dot" class:online={connected}></span>
				<span class="device-label mono">{$device}</span>
				<span class="conn-badge mono" class:live={connected}>{connected ? 'LIVE' : 'OFFLINE'}</span>
			</div>
		</div>
	</nav>

	<main class="content">
		{#if $currentJob}
			<div class="activity-bar">
				<div class="activity-info mono">
					<span class="activity-type">{$currentJob.job_type.replace('_', ' ')}</span>
					<span class="activity-clip">{$currentJob.clip_name}</span>
					{#if $currentJob.total_frames > 0}
						<span class="activity-pct">{Math.round(($currentJob.current_frame / $currentJob.total_frames) * 100)}%</span>
					{/if}
				</div>
				<div class="activity-track">
					<div
						class="activity-fill"
						style="width: {$currentJob.total_frames > 0 ? ($currentJob.current_frame / $currentJob.total_frames) * 100 : 0}%"
					></div>
				</div>
			</div>
		{/if}
		{@render children()}
	</main>
</div>

<ToastContainer />
<KeyboardHelp />

<style>
	.shell {
		display: flex;
		height: 100vh;
		overflow: hidden;
	}

	.sidebar {
		width: var(--sidebar-w);
		min-width: var(--sidebar-w);
		background: var(--surface-1);
		border-right: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		overflow: hidden;
		position: relative;
	}

	.sidebar::before {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		height: 1px;
		background: linear-gradient(90deg, transparent, var(--accent), transparent);
		opacity: 0.5;
	}

	.sidebar-top {
		display: flex;
		flex-direction: column;
	}

	.logo {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--sp-2);
		padding: var(--sp-4);
		border-bottom: 1px solid var(--border);
		transition: background 0.2s;
	}

	.logo:hover {
		background: var(--surface-2);
	}

	.logo-img {
		width: 150px;
		height: auto;
		filter: drop-shadow(0 0 4px rgba(255, 242, 3, 0.15));
	}

	.logo-product {
		font-size: 9px;
		letter-spacing: 0.2em;
		color: var(--text-tertiary);
		font-weight: 500;
	}

	.nav-links {
		display: flex;
		flex-direction: column;
		padding: var(--sp-3) var(--sp-2);
		gap: 1px;
	}

	.nav-link {
		display: flex;
		align-items: center;
		gap: var(--sp-3);
		padding: 8px var(--sp-3);
		border-radius: var(--radius-md);
		color: var(--text-secondary);
		font-size: 14px;
		font-weight: 500;
		transition: all 0.15s ease;
		position: relative;
	}

	.nav-link:hover {
		color: var(--text-primary);
		background: var(--surface-3);
	}

	.nav-link.active {
		color: var(--accent);
		background: var(--accent-muted);
	}

	.nav-link.active::before {
		content: '';
		position: absolute;
		left: 0;
		top: 50%;
		transform: translateY(-50%);
		width: 3px;
		height: 16px;
		background: var(--accent);
		border-radius: 0 3px 3px 0;
		box-shadow: 0 0 8px rgba(255, 242, 3, 0.3);
	}

	.nav-icon {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 18px;
		height: 18px;
		flex-shrink: 0;
	}

	.sidebar-bottom {
		padding: var(--sp-3) var(--sp-4) var(--sp-4);
		border-top: 1px solid var(--border);
		display: flex;
		flex-direction: column;
		gap: var(--sp-3);
	}

	.device-row {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
	}

	.device-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--state-error);
		flex-shrink: 0;
		transition: all 0.3s ease;
	}

	.device-dot.online {
		background: var(--accent);
		box-shadow: 0 0 6px rgba(255, 242, 3, 0.4);
	}

	.device-label {
		flex: 1;
		font-size: 10px;
		color: var(--text-tertiary);
	}

	.conn-badge {
		font-size: 9px;
		letter-spacing: 0.08em;
		color: var(--state-error);
		padding: 1px 5px;
		border: 1px solid currentColor;
		border-radius: 3px;
		opacity: 0.7;
	}

	.conn-badge.live {
		color: var(--accent);
		opacity: 1;
	}

	.nav-badge {
		margin-left: auto;
		min-width: 18px;
		height: 16px;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		padding: 0 4px;
		font-size: 9px;
		font-weight: 700;
		background: var(--accent);
		color: #000;
		border-radius: 8px;
	}

	.content {
		flex: 1;
		overflow-y: auto;
		overflow-x: hidden;
		background: var(--surface-0);
		display: flex;
		flex-direction: column;
	}

	.activity-bar {
		flex-shrink: 0;
		display: flex;
		align-items: center;
		gap: var(--sp-3);
		padding: 5px var(--sp-6);
		background: var(--surface-1);
		border-bottom: 1px solid var(--border);
	}

	.activity-info {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
		font-size: 11px;
		white-space: nowrap;
		flex-shrink: 0;
	}

	.activity-type {
		color: var(--accent);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.activity-clip {
		color: var(--text-secondary);
	}

	.activity-pct {
		color: var(--text-primary);
		font-weight: 600;
	}

	.activity-track {
		flex: 1;
		height: 3px;
		background: var(--surface-3);
		border-radius: 2px;
		overflow: hidden;
	}

	.activity-fill {
		height: 100%;
		background: var(--accent);
		border-radius: 2px;
		transition: width 0.3s ease-out;
		box-shadow: 0 0 6px rgba(255, 242, 3, 0.2);
	}
</style>
