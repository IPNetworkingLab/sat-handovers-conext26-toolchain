use std::{fs::read, path::PathBuf};

use clap::{Parser, builder::OsStr};
use constellation_builder::{GW_ID, build};
use convergence_simulator::{MetricChange, simulate};
use dune_core::Dune;
use futures::stream::FuturesUnordered;
use serde::Deserialize;
use tokio::{
    runtime,
    task::{self, JoinHandle},
};
use toml;
use tqdm::lib_async::tqdm_async;
use tracing::{debug, error, info, info_span, trace_span};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

use constellation::constellation::Constellation;
use handover_filter::{
    EmitHandover, Parameters as HandoverFilterParameters, load, run as handover_filter,
};

const STEP: f64 = 1f64;

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(
        long = "configuration",
        short,
        help = "Path towards the configuration file of the toolchain."
    )]
    cfg: PathBuf,
    #[arg(long)]
    gw: Option<String>,
    #[arg(
        short,
        long,
        default_value_t = false,
        help = "Save output of intermediate stages."
    )]
    save_intermediate: bool,
    #[arg(long, default_value_t = true)]
    handovers_only: bool,
    #[arg(long, default_value_t = false)]
    logging: bool,
    #[arg(long, default_value_t = false)]
    no_cache: bool,
    #[arg(short, long, default_value_t = false, help = "Enable Rust logging.")]
    enable_log: bool,
}

use convergence_simulator::SimulationCfg;

#[derive(Deserialize, Debug)]
pub struct ToolchainParameters {
    constellation_code: PathBuf,
    handover_filter: HandoverFilterParameters,
    simulations: Vec<SimulationCfg>,
}

fn main() -> Result<(), ()> {
    let cli = Cli::parse();
    if cli.enable_log {
        tracing_subscriber::fmt().init();
    }
    let _span = trace_span!("").entered();
    info!("Starting toolchain");
    let cfg = if !cli.cfg.exists() {
        Err(())
    } else if !cli.cfg.is_file() {
        Err(())
    } else if cli.cfg.extension() != Some(&OsStr::from("toml")) {
        Err(())
    } else {
        debug!("Configuration file exists. Loading it...");
        let cfg = read(&cli.cfg).map_err(|_| ())?;
        debug!("Configuration file loaded. Parsing it...");
        toml::from_slice::<ToolchainParameters>(&cfg)
            .map_err(|err| {
                error!(?err);
                ()
            })
            .map(|cfg| Ok(cfg))?
    }?;
    info!("Parameters loaded!");
    debug!(?cli, ?cfg);

    let dump = cli.save_intermediate.then_some(cli.cfg.parent().unwrap());
    info!(?dump);
    let dump = dump.and_then(|path| Some(path.to_path_buf()));

    // Stage1: Compute connetivity events for one orbital period.
    // TODO: Allow direct computation
    let span = info_span!("stage1").entered();
    let initial_constellation = Constellation::try_from(&cfg.constellation_code).unwrap();

    let filter_cfg = &cfg.handover_filter;
    if cli.no_cache {
        let mut c = initial_constellation.clone();
        let (_shell, orbital_period) = c.orbital_periods()[0];
        c.connectivity_events(orbital_period as u64, STEP, dump.clone());
    }
    let raw = load(filter_cfg)?;
    span.exit();

    // Stage2: Filter events to determine an unified connectivity events history.
    let span = info_span!("stage2").entered();
    let state = handover_filter(&raw, filter_cfg, dump.clone());
    // FIXME: Called twice but no time to solve liftime issues for now.
    let (filtered, _) = state.nest_handovers(filter_cfg.max_delay);
    span.exit();

    let rt = runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let handles: FuturesUnordered<JoinHandle<()>> = Default::default();
        for (gw, events) in filtered {
            if let Some(accept) = &cli.gw
                && accept != gw
            {
                continue;
            }
            let handover_events = events
                .iter()
                .filter(|(_, event)| {
                    if cli.handovers_only {
                        matches!(event, EmitHandover::Handover { .. })
                    } else {
                        true
                    }
                })
                .collect::<Vec<&(u64, EmitHandover)>>();
            info!(?gw, "Simulating handovers");
            // let mut bar = pbar(Some(handover_events.len()));
            for (ts, event) in handover_events {
                // Stage3: For each event, rebuild the topology.
                let mut constellation = initial_constellation.clone();
                constellation.simulate(*ts as f64);
                for shell in &constellation.shells {
                    let _span3 = info_span!("stage3", ?ts, ?event).entered();
                    let gw = constellation.get_gateway(gw.clone()).unwrap();

                    if let Ok((topo, changes)) = build(shell, &gw, *ts as f64, &event, dump.clone())
                    {
                        debug!("Topology built.");
                        _span3.exit();

                        // Stage 4: For each event, simulate the convergence for the different
                        // synchronisation methods.
                        let dune = Dune::from(topo, false);
                        let mut add: Vec<MetricChange> = Default::default();
                        let mut rm: Vec<MetricChange> = Default::default();
                        changes.iter().for_each(|&(src, dst, metric, add_event)| {
                            let change = MetricChange { src, dst, metric };
                            if add_event {
                                add.push(change);
                            } else {
                                rm.push(change);
                            }
                        });
                        let ts = ts.clone();
                        let sim_cfg = cfg.simulations.clone();
                        let dump = dump.clone();
                        handles.push(task::spawn(async move {
                            // let _span4 = info_span!("stage4", ?ts, ?event).entered();
                            simulate(dune, GW_ID, add, rm, dump, gw.get_name(), ts, sim_cfg);
                        }));
                    }
                }
            }
        }

        for handle in tqdm_async(handles) {
            let _ = handle.await;
        }
    });
    Ok(())
}
