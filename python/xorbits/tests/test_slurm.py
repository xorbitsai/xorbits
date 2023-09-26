from ..deploy.cluster import SLURMCluster


def test_header():
    with SLURMCluster(time="00:02:00", processes=4, cores=8, memory="28G") as cluster:
        assert "#SBATCH" in cluster.commands
        assert "#SBATCH -n 1" in cluster.commands
        assert "#SBATCH --cpus-per-task=8" in cluster.commands
        assert "#SBATCH --mem28G" in cluster.commands
        assert "#SBATCH -t 00:02:00" in cluster.commands
        assert "#SBATCH -p" not in cluster.commands
        # assert "#SBATCH -A" not in cluster.commands

    with SLURMCluster(
        queue="regular",
        account="XorbitsOnSlurm",
        processes=4,
        cores=8,
        memory="28G",
    ) as cluster:
        assert "#SBATCH --cpus-per-task=8" in cluster.commands
        assert "#SBATCH --mem=28G" in cluster.commands
        assert "#SBATCH -t " in cluster.commands
        assert "#SBATCH -A XorbitsOnSlurm" in cluster.commands
        assert "#SBATCH --partion regular" in cluster.commands
