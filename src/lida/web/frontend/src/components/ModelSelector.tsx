import React, { useEffect, useState } from 'react';

interface Model {
  name: string;
}

interface ModelProvider {
  name: string;
  models: Model[];
}

interface ModelsResponse {
  status: boolean;
  data: Record<string, ModelProvider>;
  default: string;
  message: string;
}

interface ModelSelectorProps {
  onModelChange: (provider: string, model: string) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ onModelChange }) => {
  const [providers, setProviders] = useState<Record<string, ModelProvider>>({});
  const [selectedProvider, setSelectedProvider] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetch('/api/v1/models')
      .then((res) => res.json())
      .then((data: ModelsResponse) => {
        if (data.status) {
          setProviders(data.data);
          const defaultProviderName = data.default;
          // Verify default provider exists in response, otherwise pick first
          const providerKey = data.data[defaultProviderName] ? defaultProviderName : Object.keys(data.data)[0];
          
          if (providerKey) {
            setSelectedProvider(providerKey);
            const models = data.data[providerKey].models;
            if (models && models.length > 0) {
                const firstModel = models[0].name;
                setSelectedModel(firstModel);
                onModelChange(providerKey, firstModel);
            }
          }
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch models:', err);
        setLoading(false);
      });
  }, [onModelChange]);

  const handleProviderChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newProvider = e.target.value;
    setSelectedProvider(newProvider);
    const models = providers[newProvider]?.models || [];
    if (models.length > 0) {
      const firstModel = models[0].name;
      setSelectedModel(firstModel);
      onModelChange(newProvider, firstModel);
    } else {
        setSelectedModel('');
        onModelChange(newProvider, '');
    }
  };

  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);
    onModelChange(selectedProvider, newModel);
  };

  if (loading) {
    return <div>Loading models...</div>;
  }

  const currentModels = providers[selectedProvider]?.models || [];

  return (
    <div className="model-selector">
      <div className="form-group">
        <label htmlFor="provider-select">Provider:</label>
        <select id="provider-select" value={selectedProvider} onChange={handleProviderChange}>
          {Object.keys(providers).map((key) => (
            <option key={key} value={key}>
              {key}
            </option>
          ))}
        </select>
      </div>

      {currentModels.length > 0 && (
        <div className="form-group">
          <label htmlFor="model-select">Model:</label>
          <select id="model-select" value={selectedModel} onChange={handleModelChange}>
            {currentModels.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
};

