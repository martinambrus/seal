import React from 'react';

export const ResultsPanel = ({ results, currentView, onRegionClick }) => {
    if (!results || !results.success) {
        return (
            <div className="results-panel empty">
                <p>No results yet. Upload images and click Compare.</p>
            </div>
        );
    }

    const renderMethodSummary = (methodName, methodData) => {
        if (!methodData) return null;

        const numRegions = methodData.num_regions || 0;
        const confidence = Math.round((methodData.overall_score || 0) * 100);
        const executionTime = methodData.execution_time?.toFixed(2) || 'N/A';

        return (
            <div key={methodName} className="method-summary">
                <div className="method-summary-header">
                    <span className="method-name">{formatMethodName(methodName)}</span>
                    <span className={`status-badge ${numRegions > 0 ? 'detected' : 'clean'}`}>
                        {numRegions > 0 ? '✓ Differences Found' : '○ No Differences'}
                    </span>
                </div>
                <div className="method-summary-stats">
                    <div className="stat">
                        <label>Regions:</label>
                        <span>{numRegions}</span>
                    </div>
                    <div className="stat">
                        <label>Confidence:</label>
                        <span>{confidence}%</span>
                    </div>
                    <div className="stat">
                        <label>Time:</label>
                        <span>{executionTime}s</span>
                    </div>
                </div>
            </div>
        );
    };

    const renderRegionDetails = () => {
        let regions = [];

        if (currentView === 'combined' && results.results.ensemble) {
            regions = results.results.ensemble.regions || [];
        } else if (results.results[currentView]) {
            regions = results.results[currentView].regions || [];
        }

        if (regions.length === 0) {
            return <p className="no-regions">No differences detected in this view.</p>;
        }

        return (
            <div className="region-list">
                <h4>Detected Regions ({regions.length})</h4>
                {regions.map((region, index) => (
                    <div
                        key={index}
                        className="region-item clickable"
                        onClick={() => onRegionClick && onRegionClick(region)}
                    >
                        <div className="region-header">
                            <span className="region-number">Region {index + 1}</span>
                            <span className="region-confidence">
                                {Math.round((region.confidence || 0) * 100)}% confidence
                            </span>
                        </div>
                        <div className="region-details">
                            <span>Position: ({region.bbox[0]}, {region.bbox[1]})</span>
                            <span>Size: {region.bbox[2]}×{region.bbox[3]}px</span>
                            <span>Area: {region.area_pixels}px²</span>
                        </div>
                        {region.detected_by && region.detected_by.length > 0 && (
                            <div className="detected-by">
                                Detected by: {region.detected_by.map(formatMethodName).join(', ')}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        );
    };

    const formatMethodName = (name) => {
        const names = {
            'ssim': 'SSIM',
            'pixel_diff': 'Pixel Diff',
            'color': 'Color Analysis',
            'edge': 'Edge Detection',
            'ocr': 'OCR',
            'siamese': 'Siamese Net',
            'cnn': 'CNN Detector',
            'autoencoder': 'Autoencoder',
            'combined': 'Ensemble',
            'ensemble': 'Ensemble'
        };
        return names[name] || name;
    };

    const handleExportJSON = () => {
        const dataStr = JSON.stringify(results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `comparison-results-${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="results-panel">
            <div className="results-header">
                <h2>Results Summary</h2>
                <div className="alignment-quality">
                    Alignment Quality: <strong>{Math.round(results.alignment_quality || 0)}%</strong>
                </div>
            </div>

            <div className="methods-summary">
                <h3>Methods Overview</h3>
                {Object.entries(results.results || {}).map(([name, data]) =>
                    renderMethodSummary(name, data)
                )}
            </div>

            <div className="current-view-details">
                <h3>Current View: {formatMethodName(currentView)}</h3>
                {renderRegionDetails()}
            </div>

            <div className="export-actions">
                <button className="export-button" onClick={handleExportJSON}>
                    📥 Export Report (JSON)
                </button>
                <button className="export-button" disabled>
                    🖼️ Export Images
                </button>
            </div>
        </div>
    );
};
