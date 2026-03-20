<script lang="ts">
	import type { InferenceParams, OutputConfig } from '$lib/api';

	let {
		params = $bindable<InferenceParams>({
			input_is_linear: false,
			despill_strength: 1.0,
			auto_despeckle: true,
			despeckle_size: 400,
			refiner_scale: 1.0,
		}),
		outputConfig = $bindable<OutputConfig>({
			fg_enabled: true,
			fg_format: 'exr',
			matte_enabled: true,
			matte_format: 'exr',
			comp_enabled: true,
			comp_format: 'png',
			processed_enabled: true,
			processed_format: 'exr',
		}),
	}: {
		params: InferenceParams;
		outputConfig: OutputConfig;
	} = $props();

	function clamp(val: number, min: number, max: number): number {
		return Math.min(max, Math.max(min, val));
	}
</script>

<div class="form-panel">
	<div class="form-section">
		<h3 class="section-title mono">INFERENCE PARAMS</h3>

		<label class="field">
			<span class="field-label">Despill Strength</span>
			<div class="slider-row">
				<input type="range" min="0" max="1" step="0.01" bind:value={params.despill_strength} class="slider" />
				<input
					type="number"
					min="0" max="1" step="0.01"
					class="num-input mono"
					value={params.despill_strength.toFixed(2)}
					onchange={(e) => { params.despill_strength = clamp(parseFloat((e.target as HTMLInputElement).value) || 0, 0, 1); }}
				/>
			</div>
		</label>

		<label class="field">
			<span class="field-label">Refiner Scale</span>
			<div class="slider-row">
				<input type="range" min="0" max="2" step="0.01" bind:value={params.refiner_scale} class="slider" />
				<input
					type="number"
					min="0" max="2" step="0.01"
					class="num-input mono"
					value={params.refiner_scale.toFixed(2)}
					onchange={(e) => { params.refiner_scale = clamp(parseFloat((e.target as HTMLInputElement).value) || 0, 0, 2); }}
				/>
			</div>
		</label>

		<label class="field">
			<span class="field-label">Despeckle Size</span>
			<div class="slider-row">
				<input type="range" min="1" max="2000" step="1" bind:value={params.despeckle_size} class="slider" />
				<input
					type="number"
					min="1" max="2000" step="1"
					class="num-input mono"
					value={params.despeckle_size}
					onchange={(e) => { params.despeckle_size = clamp(parseInt((e.target as HTMLInputElement).value) || 1, 1, 2000); }}
				/>
			</div>
		</label>

		<div class="toggle-group">
			<label class="toggle-field">
				<input type="checkbox" bind:checked={params.auto_despeckle} class="toggle" />
				<span>Auto Despeckle</span>
			</label>
			<label class="toggle-field">
				<input type="checkbox" bind:checked={params.input_is_linear} class="toggle" />
				<span>Input is Linear</span>
			</label>
		</div>
	</div>

	<div class="form-section">
		<h3 class="section-title mono">OUTPUT PASSES</h3>

		<div class="output-grid">
			<div class="output-row">
				<label class="toggle-field">
					<input type="checkbox" bind:checked={outputConfig.fg_enabled} class="toggle" />
					<span>FG</span>
				</label>
				<select bind:value={outputConfig.fg_format} class="format-select mono" disabled={!outputConfig.fg_enabled}>
					<option value="exr">EXR</option>
					<option value="png">PNG</option>
				</select>
			</div>
			<div class="output-row">
				<label class="toggle-field">
					<input type="checkbox" bind:checked={outputConfig.matte_enabled} class="toggle" />
					<span>Matte</span>
				</label>
				<select bind:value={outputConfig.matte_format} class="format-select mono" disabled={!outputConfig.matte_enabled}>
					<option value="exr">EXR</option>
					<option value="png">PNG</option>
				</select>
			</div>
			<div class="output-row">
				<label class="toggle-field">
					<input type="checkbox" bind:checked={outputConfig.comp_enabled} class="toggle" />
					<span>Comp</span>
				</label>
				<select bind:value={outputConfig.comp_format} class="format-select mono" disabled={!outputConfig.comp_enabled}>
					<option value="png">PNG</option>
					<option value="exr">EXR</option>
				</select>
			</div>
			<div class="output-row">
				<label class="toggle-field">
					<input type="checkbox" bind:checked={outputConfig.processed_enabled} class="toggle" />
					<span>Processed</span>
				</label>
				<select bind:value={outputConfig.processed_format} class="format-select mono" disabled={!outputConfig.processed_enabled}>
					<option value="exr">EXR</option>
					<option value="png">PNG</option>
				</select>
			</div>
		</div>
	</div>
</div>

<style>
	.form-panel {
		display: flex;
		flex-direction: column;
		gap: var(--sp-6);
	}

	.form-section {
		display: flex;
		flex-direction: column;
		gap: var(--sp-3);
	}

	.section-title {
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.12em;
		color: var(--accent);
		padding-bottom: var(--sp-2);
		border-bottom: 1px solid var(--border);
	}

	.field {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.field-label {
		font-size: 12px;
		color: var(--text-secondary);
	}

	.slider-row {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
	}

	.slider {
		flex: 1;
		-webkit-appearance: none;
		appearance: none;
		height: 3px;
		background: var(--surface-4);
		border-radius: 2px;
		outline: none;
		cursor: pointer;
	}

	.slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--accent);
		cursor: pointer;
		border: 2px solid var(--surface-2);
	}

	.slider::-moz-range-thumb {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--accent);
		cursor: pointer;
		border: 2px solid var(--surface-2);
	}

	.num-input {
		width: 52px;
		padding: 3px 6px;
		font-size: 11px;
		text-align: right;
		background: var(--surface-3);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text-primary);
		outline: none;
		flex-shrink: 0;
	}

	.num-input:focus {
		border-color: var(--accent);
	}

	/* Hide spin buttons for cleaner look */
	.num-input::-webkit-inner-spin-button,
	.num-input::-webkit-outer-spin-button {
		-webkit-appearance: none;
		margin: 0;
	}
	.num-input { -moz-appearance: textfield; }

	.toggle-group {
		display: flex;
		flex-direction: column;
		gap: var(--sp-2);
	}

	.toggle-field {
		display: flex;
		align-items: center;
		gap: var(--sp-2);
		font-size: 12px;
		color: var(--text-secondary);
		cursor: pointer;
	}

	.toggle {
		-webkit-appearance: none;
		appearance: none;
		width: 28px;
		height: 14px;
		border-radius: 7px;
		background: var(--surface-4);
		position: relative;
		cursor: pointer;
		transition: background 0.15s;
		flex-shrink: 0;
	}

	.toggle::after {
		content: '';
		position: absolute;
		top: 2px;
		left: 2px;
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--text-tertiary);
		transition: transform 0.15s, background 0.15s;
	}

	.toggle:checked {
		background: var(--accent-muted);
	}

	.toggle:checked::after {
		transform: translateX(14px);
		background: var(--accent);
	}

	.output-grid {
		display: flex;
		flex-direction: column;
		gap: var(--sp-2);
	}

	.output-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--sp-3);
	}

	.format-select {
		padding: 2px 6px;
		font-size: 10px;
		background: var(--surface-3);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text-secondary);
		cursor: pointer;
	}

	.format-select:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
</style>
