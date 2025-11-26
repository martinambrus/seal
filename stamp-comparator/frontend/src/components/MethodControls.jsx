import React from 'react';
import ReactSlider from 'react-slider';

const METHODS_CONFIG = [
    {
        category: 'Traditional CV',
        methods: [
            {
                id: 'ssim', name: 'SSIM', description: 'Structural similarity',
                min: 0.5, max: 1.0, step: 0.01, default: 0.95
            },
            {
                id: 'pixel_diff', name: 'Pixel Difference', description: 'Raw pixel comparison',
                min: 0.0, max: 1.0, step: 0.01, default: 0.20
            },
            {
                id: 'color', name: 'Color Analysis', description: 'RGB channel differences',
                min: 0.0, max: 1.0, step: 0.01, default: 0.25
            },
            {
                id: 'edge', name: 'Edge Detection', description: 'Canny edge comparison',
                min: 0.0, max: 1.0, step: 0.01, default: 0.15
            },
            {
                id: 'ocr', name: 'OCR Comparison', description: 'Text detection & matching',
                min: 0.0, max: 1.0, step: 0.01, default: 0.80
            }
        ]
    },
    {
        category: 'Machine Learning',
        methods: [
            {
                id: 'siamese', name: 'Siamese Network', description: 'Neural similarity',
                min: 0.0, max: 1.0, step: 0.01, default: 0.90
            },
            {
                id: 'cnn', name: 'CNN Detector', description: 'Deep learning localization',
                min: 0.0, max: 1.0, step: 0.01, default: 0.75
            },
            {
                id: 'autoencoder', name: 'Autoencoder', description: 'Anomaly detection',
                min: 0.0, max: 1.0, step: 0.01, default: 0.30
            }
        ]
    }
];

export const MethodControls = ({
    enabledMethods,
    thresholds,
    onMethodToggle,
    onThresholdChange,
    processingStatus = {}
}) => {
    const renderMethod = (method) => {
        const isEnabled = enabledMethods[method.id];
        const threshold = thresholds[method.id];
        const status = processingStatus[method.id];

        return (
            <div key={method.id} className={`method-control ${isEnabled ? 'enabled' : 'disabled'}`}>
                <div className="method-header">
                    <label className="method-toggle">
                        <input
                            type="checkbox"
                            checked={isEnabled}
                            onChange={(e) => onMethodToggle(method.id, e.target.checked)}
                        />
                        <span className="method-name">{method.name}</span>
                        {status && <span className={`status-badge ${status}`}>{status}</span>}
                    </label>
                    <span className="method-description">{method.description}</span>
                </div>

                {isEnabled && (
                    <div className="threshold-control">
                        <label className="threshold-label">
                            Threshold: <strong>{threshold?.toFixed(2) || method.default.toFixed(2)}</strong>
                        </label>
                        <ReactSlider
                            className="threshold-slider"
                            thumbClassName="slider-thumb"
                            trackClassName="slider-track"
                            value={threshold || method.default}
                            min={method.min}
                            max={method.max}
                            step={method.step}
                            onChange={(value) => onThresholdChange(method.id, value)}
                        />
                        <div className="threshold-range">
                            <span>{method.min.toFixed(2)}</span>
                            <span>{method.max.toFixed(2)}</span>
                        </div>
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="method-controls">
            <h2>Detection Methods</h2>

            {METHODS_CONFIG.map((category) => (
                <div key={category.category} className="method-category">
                    <h3>{category.category}</h3>
                    <div className="methods-list">
                        {category.methods.map(renderMethod)}
                    </div>
                </div>
            ))}
        </div>
    );
};
