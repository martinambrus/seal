import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const api = {
    // Compare two stamp images
    compareStamps: async (referenceFile, testFile, config) => {
        const formData = new FormData();
        formData.append('reference_image', referenceFile);
        formData.append('test_image', testFile);
        formData.append('config', JSON.stringify(config));

        const response = await axios.post(`${API_BASE_URL}/compare`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });

        return response.data;
    },

    // Get default configuration
    getDefaultConfig: async () => {
        const response = await axios.get(`${API_BASE_URL}/config/default`);
        return response.data;
    },

    // Save configuration
    saveConfig: async (name, config) => {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('config_json', JSON.stringify(config));

        const response = await axios.post(`${API_BASE_URL}/config/save`, formData);
        return response.data;
    },

    // Load configuration
    loadConfig: async (name) => {
        const response = await axios.get(`${API_BASE_URL}/config/load/${name}`);
        return response.data;
    },

    // Health check
    healthCheck: async () => {
        const response = await axios.get('http://localhost:8000/health');
        return response.data;
    }
};
