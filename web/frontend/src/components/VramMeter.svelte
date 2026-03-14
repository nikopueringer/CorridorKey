<script lang="ts">
	import { vram } from '$lib/stores/system';

	let info = $derived($vram);
	let pct = $derived(info && info.total > 0 ? (info.allocated / info.total) * 100 : 0);
	let allocGb = $derived(info ? info.allocated.toFixed(1) : '0.0');
	let totalGb = $derived(info ? info.total.toFixed(1) : '0.0');
</script>

<div class="vram">
	<div class="vram-header">
		<span class="vram-title mono">VRAM</span>
		{#if info && info.available}
			<span class="vram-readout mono">{allocGb}<span class="dim">/{totalGb} GB</span></span>
		{:else}
			<span class="vram-readout mono dim">N/A</span>
		{/if}
	</div>
	<div class="vram-track">
		<div
			class="vram-fill"
			class:warn={pct > 80}
			class:crit={pct > 95}
			style="width: {pct}%"
		></div>
		<!-- Tick marks at 25%, 50%, 75% -->
		<div class="tick" style="left: 25%"></div>
		<div class="tick" style="left: 50%"></div>
		<div class="tick" style="left: 75%"></div>
	</div>
	{#if info?.name}
		<span class="vram-gpu mono">{info.name}</span>
	{/if}
</div>

<style>
	.vram {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}

	.vram-header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}

	.vram-title {
		font-size: 9px;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		color: var(--accent);
		font-weight: 600;
	}

	.vram-readout {
		font-size: 10px;
		color: var(--text-primary);
	}

	.dim {
		color: var(--text-tertiary);
	}

	.vram-track {
		height: 5px;
		background: var(--surface-3);
		border-radius: 3px;
		overflow: hidden;
		position: relative;
		box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.3);
	}

	.vram-fill {
		height: 100%;
		background: linear-gradient(90deg, #007aaa, var(--secondary));
		border-radius: 3px;
		transition: width 0.5s ease-out, background 0.3s;
	}

	.vram-fill.warn {
		background: linear-gradient(90deg, var(--secondary), var(--state-raw));
	}

	.vram-fill.crit {
		background: linear-gradient(90deg, var(--state-raw), var(--state-error));
		box-shadow: 0 0 6px rgba(255, 82, 82, 0.3);
	}

	.tick {
		position: absolute;
		top: 0;
		width: 1px;
		height: 100%;
		background: rgba(255, 255, 255, 0.06);
	}

	.vram-gpu {
		font-size: 9px;
		color: var(--text-tertiary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
</style>
