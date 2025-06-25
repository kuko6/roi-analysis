"""
Basic import test for roi-analyser library.
"""

import pytest


def test_import_roi_analyser():
    """Test that RoiAnalyser can be imported from the package."""
    from roi_analyser import RoiAnalyser
    assert RoiAnalyser is not None


def test_package_version():
    """Test that package version is accessible."""
    import roi_analyser
    assert hasattr(roi_analyser, '__version__')
    assert roi_analyser.__version__ == "0.1.0"


def test_package_all():
    """Test that __all__ is properly defined."""
    import roi_analyser
    assert hasattr(roi_analyser, '__all__')
    assert 'RoiAnalyser' in roi_analyser.__all__


def test_roi_analyser_class_exists():
    """Test that RoiAnalyser class has expected attributes."""
    from roi_analyser import RoiAnalyser
    
    # Check that it's a class
    assert isinstance(RoiAnalyser, type)
    
    # Check that it has key methods
    expected_methods = [
        'create_binary_mask',
        'define_clusters', 
        'plot_contours',
        'build_histogram',
        'run_analysis'
    ]
    
    for method in expected_methods:
        assert hasattr(RoiAnalyser, method), f"RoiAnalyser missing method: {method}"