from typing import Dict
import click

from cli_utils import create_model_and_update_fn, DistanceChoices, DistanceType


@click.group("covertree")
@click.option("-s", "--spatial-resolution", type=float, required=True)
@click.option("-d", "--distance-type", type=DistanceChoices, default="euclidean")
@click.pass_context
def covertree(ctx: click.Context, spatial_resolution: float, distance_type: DistanceType):
    ctx_obj: Dict = ctx.obj
    common_ctx: Dict = ctx_obj["common_ctx"]

    clustering_type = "covertree"
    clustering_kwargs = {"spatial_resolution": spatial_resolution}

    model, update_fn = create_model_and_update_fn(
        common_ctx["model_class_fn"],
        common_ctx["dataset"].train,
        clustering_type=clustering_type,
        distance_type=distance_type,
        use_jit=common_ctx["jit"],
        clustering_kwargs=clustering_kwargs,
    )

    ctx_obj["ip_ctx"] = dict(
        model=model,
        update_fn=update_fn,
        clustering_type=clustering_type,
        clustering_kwargs=clustering_kwargs,
        distance_type=distance_type,
    )


@click.group("kmeans")
@click.option("-m", "--max-num-ip", type=int, required=True)
@click.option("-d", "--distance-type", type=DistanceChoices, default="euclidean")
@click.pass_context
def kmeans(ctx: click.Context, max_num_ip: int, distance_type: DistanceType):
    ctx_obj: Dict = ctx.obj
    common_ctx: Dict = ctx_obj["common_ctx"]

    clustering_type = "kmeans"
    clustering_kwargs = {"num_inducing_points": max_num_ip}

    model, update_fn = create_model_and_update_fn(
        common_ctx["model_class_fn"],
        common_ctx["dataset"].train,
        clustering_type=clustering_type,
        distance_type=distance_type,
        use_jit=common_ctx["jit"],
        clustering_kwargs=clustering_kwargs,
    )

    ctx_obj["ip_ctx"] = dict(
        model=model,
        update_fn=update_fn,
        clustering_type=clustering_type,
        clustering_kwargs=clustering_kwargs,
        distance_type=distance_type,
    )


@click.group("oips")
@click.option("-r", "--rho", type=float, required=True)
@click.option("-m", "--max-num-ip", type=int)
@click.option("-d", "--distance-type", type=DistanceChoices, default="euclidean")
@click.pass_context
def oips(ctx: click.Context, rho: float, max_num_ip: int, distance_type: DistanceType):
    ctx_obj: Dict = ctx.obj
    common_ctx: Dict = ctx_obj["common_ctx"]

    clustering_type = "oips"
    clustering_kwargs = {"rho": rho, "max_points": max_num_ip}

    model, update_fn = create_model_and_update_fn(
        common_ctx["model_class_fn"],
        common_ctx["dataset"].train,
        clustering_type=clustering_type,
        distance_type=distance_type,
        use_jit=common_ctx["jit"],
        clustering_kwargs=clustering_kwargs,
    )

    ctx_obj["ip_ctx"] = dict(
        model=model,
        update_fn=update_fn,
        clustering_type=clustering_type,
        clustering_kwargs=clustering_kwargs,
        distance_type=distance_type,
    )


@click.group("uniform")
@click.option("-m", "--max-num-ip", type=int, required=True)
@click.option("-d", "--distance-type", type=DistanceChoices, default="euclidean")
@click.pass_context
def uniform(ctx: click.Context, max_num_ip: int, distance_type: DistanceType):
    ctx_obj: Dict = ctx.obj
    common_ctx: Dict = ctx_obj["common_ctx"]

    clustering_type = "uniform"
    clustering_kwargs = {"max_points": max_num_ip}

    model, update_fn = create_model_and_update_fn(
        common_ctx["model_class_fn"],
        common_ctx["dataset"].train,
        clustering_type=clustering_type,
        distance_type=distance_type,
        use_jit=common_ctx["jit"],
        clustering_kwargs=clustering_kwargs,
    )

    ctx_obj["ip_ctx"] = dict(
        model=model,
        update_fn=update_fn,
        clustering_type=clustering_type,
        clustering_kwargs=clustering_kwargs,
        distance_type=distance_type,
    )
