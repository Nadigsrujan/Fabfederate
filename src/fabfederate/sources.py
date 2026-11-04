"""Explicit source dispatch; datasets retain separate units and provenance."""


def load_source(config):
    if config.get("dataset", "nasa") == "nasa":
        from fabfederate.data.nasa_loader import load_nasa
        return load_nasa(config["data_path"])
    if config.get("dataset") == "phm2010":
        from fabfederate.data.phm_loader import load_phm
        return load_phm(config["data_path"], edge_reduction=config.get("phm_edge_reduction"))
    raise ValueError("Dataset must be nasa or phm2010")
