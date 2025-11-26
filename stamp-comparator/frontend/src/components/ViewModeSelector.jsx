import React from 'react';

const VIEW_OPTIONS = [
    { id: 'combined', name: 'Combined (Ensemble)', icon: '🔍' },
    { id: 'ssim', name: 'SSIM', icon: '📊', category: 'cv' },
    { id: 'pixel_diff', name: 'Pixel Difference', icon: '🔲', category: 'cv' },
    { id: 'color', name: 'Color Analysis', icon: '🎨', category: 'cv' },
    { id: 'edge', name: 'Edge Detection', icon: '📐', category: 'cv' },
    { id: 'ocr', name: 'OCR', icon: '📝', category: 'cv' },
    { id: 'siamese', name: 'Siamese Network', icon: '🤖', category: 'ml' },
    { id: 'cnn', name: 'CNN Detector', icon: '🧠', category: 'ml' },
    { id: 'autoencoder', name: 'Autoencoder', icon: '⚡', category: 'ml' }
];

export const ViewModeSelector = ({
    currentView,
    onViewChange,
    availableMethods = [],
    methodStats = {}
}) => {
    const renderOption = (option) => {
        const isAvailable = option.id === 'combined' || availableMethods.includes(option.id);
        const isCurrent = currentView === option.id;
        const stats = methodStats[option.id];

        return (
            <button
                key={option.id}
                className={`view-option ${isCurrent ? 'active' : ''} ${!isAvailable ? 'disabled' : ''}`}
                onClick={() => isAvailable && onViewChange(option.id)}
                disabled={!isAvailable}
            >
                <span className="option-icon">{option.icon}</span>
                <div className="option-content">
                    <span className="option-name">{option.name}</span>
                    {stats && (
                        <span className="option-stats">
                            {stats.regions} regions · {stats.confidence}% confidence
                        </span>
                    )}
                </div>
                {!isAvailable && <span className="unavailable-badge">Not Run</span>}
            </button>
        );
    };

    return (
        <div className="view-mode-selector">
            <h3>View Results</h3>

            <div className="view-options-list">
                {/* Combined view always first */}
                {renderOption(VIEW_OPTIONS[0])}

                <div className="separator">Computer Vision Methods</div>
                {VIEW_OPTIONS.filter(opt => opt.category === 'cv').map(renderOption)}

                <div className="separator">Machine Learning Methods</div>
                {VIEW_OPTIONS.filter(opt => opt.category === 'ml').map(renderOption)}
            </div>
        </div>
    );
};
