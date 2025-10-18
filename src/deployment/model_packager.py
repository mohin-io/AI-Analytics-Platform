"""
Model Packager

Package models with dependencies for deployment.
"""

import pickle
import json
import zipfile
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib


class ModelPackager:
    """
    Package trained models with metadata and dependencies.

    Creates self-contained deployment packages.
    """

    def __init__(self):
        """Initialize model packager."""
        self.package_version = "1.0.0"

    def package_model(self, model: Any, metadata: Dict[str, Any],
                     output_path: str, include_dependencies: bool = True) -> str:
        """
        Package model for deployment.

        Args:
            model: Trained model
            metadata: Model metadata
            output_path: Output package path
            include_dependencies: Include dependency info

        Returns:
            Path to created package
        """
        # Create temporary directory structure
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        try:
            # Save model
            model_path = os.path.join(temp_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            # Save metadata
            package_metadata = {
                'model_metadata': metadata,
                'package_version': self.package_version,
                'created_at': datetime.now().isoformat(),
                'model_hash': self._calculate_model_hash(model)
            }

            if include_dependencies:
                package_metadata['dependencies'] = self._get_dependencies()

            metadata_path = os.path.join(temp_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(package_metadata, f, indent=2)

            # Create README
            readme_path = os.path.join(temp_dir, 'README.md')
            with open(readme_path, 'w') as f:
                f.write(self._generate_readme(package_metadata))

            # Create zip package
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)

        return output_path

    def load_package(self, package_path: str) -> tuple[Any, Dict[str, Any]]:
        """
        Load model from package.

        Args:
            package_path: Path to package file

        Returns:
            Tuple of (model, metadata)
        """
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        try:
            # Extract package
            with zipfile.ZipFile(package_path, 'r') as zipf:
                zipf.extractall(temp_dir)

            # Load model
            model_path = os.path.join(temp_dir, 'model.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Load metadata
            metadata_path = os.path.join(temp_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return model, metadata

        finally:
            shutil.rmtree(temp_dir)

    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of model for versioning."""
        model_bytes = pickle.dumps(model)
        return hashlib.sha256(model_bytes).hexdigest()

    def _get_dependencies(self) -> List[str]:
        """Get list of dependencies."""
        import pkg_resources

        dependencies = []
        for package in ['numpy', 'scikit-learn', 'pandas']:
            try:
                version = pkg_resources.get_distribution(package).version
                dependencies.append(f"{package}=={version}")
            except:
                pass

        return dependencies

    def _generate_readme(self, metadata: Dict[str, Any]) -> str:
        """Generate README for package."""
        readme = f"""# ML Model Package

## Package Information
- Version: {metadata['package_version']}
- Created: {metadata['created_at']}
- Model Hash: {metadata['model_hash'][:16]}...

## Model Metadata
{json.dumps(metadata.get('model_metadata', {}), indent=2)}

## Usage
```python
from src.deployment import ModelPackager

packager = ModelPackager()
model, metadata = packager.load_package('path/to/package.zip')

# Make predictions
predictions = model.predict(X_test)
```

## Dependencies
"""
        if 'dependencies' in metadata:
            for dep in metadata['dependencies']:
                readme += f"- {dep}\n"

        return readme
