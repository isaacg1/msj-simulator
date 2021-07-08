use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::weighted_alias::WeightedAliasIndex;
use rand_distr::Exp;

use std::collections::HashSet;

use std::f64::INFINITY;
const EPSILON: f64 = 1e-7;

#[derive(Debug)]
struct Job {
    arrival_time: f64,
    remaining_service: f64,
    num_servers: u64,
}

#[derive(Debug, Clone)]
struct Dist {
    servers_dists: Vec<(u64, Exp<f64>)>,
    servers_rates_probs: Vec<(u64, f64, f64)>,
    index_dist: WeightedAliasIndex<f64>,
}
impl Dist {
    fn new(servers_rates_probs: &Vec<(u64, f64, f64)>) -> Dist {
        let weights = servers_rates_probs.iter().map(|(_, _, p)| *p).collect();
        let index_dist = WeightedAliasIndex::new(weights).unwrap();
        let servers_dists = servers_rates_probs
            .iter()
            .map(|(s, mu, _)| (*s, Exp::new(*mu).unwrap()))
            .collect();
        Dist {
            servers_rates_probs: servers_rates_probs.clone(),
            servers_dists,
            index_dist,
        }
    }
    fn sample<R: Rng>(&self, rng: &mut R) -> (u64, f64) {
        let index = self.index_dist.sample(rng);
        let (servers, dist) = self.servers_dists[index];
        let service = dist.sample(rng);
        (servers, service)
    }
    // Measured in units of servers seconds
    fn mean_size(&self) -> f64 {
        self.servers_rates_probs
            .iter()
            .map(|(n_i, mu_i, p_i)| *n_i as f64 * p_i / mu_i)
            .sum()
    }
    fn mean_size_excess(&self) -> f64 {
        let mean_sq: f64 = self
            .servers_rates_probs
            .iter()
            .map(|(n_i, mu_i, p_i)| p_i * n_i.pow(2) as f64 * 2.0 / mu_i.powi(2))
            .sum();
        mean_sq / (2.0 * self.mean_size())
    }
    // Measured in units of seconds
    fn mean_service_time(&self) -> f64 {
        self.servers_rates_probs
            .iter()
            .map(|(_, mu_i, p_i)| p_i / mu_i)
            .sum()
    }
}

#[derive(Debug)]
enum Policy {
    ServerFilling,
    FCFS,
    BestFit,
    FirstFit,
    EASYBackfilling,
    ConservativeBackfilling,
    MaxWeight,
    RandomizedTimers,
    StallingMaxWeight(f64),
    LeastServersFirst,
    MostServersFirst,
}

impl Policy {
    fn serve(&self, queue: &Vec<Job>, num_servers: u64) -> Vec<usize> {
        let indices = match self {
            Policy::ServerFilling => {
                let mut front_size = 0;
                let mut servers_occupied_by_front = 0;
                for (i, job) in queue.iter().enumerate() {
                    servers_occupied_by_front += job.num_servers;
                    front_size = i + 1;
                    if servers_occupied_by_front >= num_servers {
                        break;
                    }
                }
                let mut front: Vec<usize> = (0..front_size).collect();
                front.sort_by_key(|&i| num_servers - queue[i].num_servers);
                let mut service = vec![];
                let mut servers_occupied = 0;
                for i in front {
                    servers_occupied += queue[i].num_servers;
                    if servers_occupied <= num_servers {
                        service.push(i);
                        if servers_occupied == num_servers {
                            break;
                        }
                    } else {
                        unreachable!()
                    }
                }
                service
            }
            Policy::FCFS => {
                let mut servers_occupied = 0;
                let mut service = vec![];
                for (i, job) in queue.iter().enumerate() {
                    if servers_occupied + job.num_servers <= num_servers {
                        servers_occupied += job.num_servers;
                        service.push(i);
                    } else {
                        break;
                    }
                }
                service
            }
            Policy::FirstFit => {
                let mut servers_occupied = 0;
                let mut service = vec![];
                for (i, job) in queue.iter().enumerate() {
                    if servers_occupied + job.num_servers <= num_servers {
                        servers_occupied += job.num_servers;
                        service.push(i);
                        if servers_occupied == num_servers {
                            break
                        }
                    }
                }
                service
            }
            _ => todo!(),
        };
        let num_servers_occupied: u64 = indices.iter().map(|&i| queue[i].num_servers).sum();
        assert!(num_servers_occupied <= num_servers);
        indices
    }
}

#[derive(Default, Debug)]
struct Recording {
    total_response_time: f64,
    num_completed: u64,
    total_queueing_time: f64,
    num_served: u64,
    total_monotonic_queueing_time: f64,
    num_monotonic_served: u64,
}
impl Recording {
    fn output(&self) -> Results {
        Results {
            mean_response_time: self.total_response_time / self.num_completed as f64,
            mean_queueing_time: self.total_queueing_time / self.num_served as f64,
            mean_monotonic_queueing_time: self.total_monotonic_queueing_time
                / self.num_monotonic_served as f64,
        }
    }
}

struct Results {
    mean_response_time: f64,
    mean_queueing_time: f64,
    mean_monotonic_queueing_time: f64,
}

fn simulate(
    policy: &Policy,
    dist: Dist,
    num_servers: u64,
    rho: f64,
    num_jobs: u64,
    seed: u64,
) -> Results {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut queue: Vec<Job> = Vec::new();
    let mut time: f64 = 0.0;
    let mean_size = dist.mean_size();
    let mean_absolute_size = mean_size / num_servers as f64;
    let lambda = rho / mean_absolute_size;
    let arrival_dist = Exp::new(lambda).unwrap();
    let mut next_arrival_time = time + arrival_dist.sample(&mut rng);
    let mut recording = Recording::default();
    let mut num_completed = 0;
    let mut num_arrivals = 0;
    let mut served_and_incomplete: HashSet<N64> = HashSet::new();
    let mut newest_served = None;
    let debug = false;
    while num_completed < num_jobs {
        let service = policy.serve(&queue, num_servers);
        if debug {
            dbg!(
                &queue,
                &service,
            );
            std::io::stdin().read_line(&mut String::new()).unwrap();
        }
        let min_service: f64 = service
            .iter()
            .map(|&index| queue[index].remaining_service)
            .min_by_key(|&f| n64(f))
            .unwrap_or(INFINITY);
        for &index in &service {
            if newest_served.map_or(true, |newest| index > newest) {
                for i in newest_served.map_or(0, |newest| newest + 1)..=index {
                    let monotonic_queueing = time - queue[i].arrival_time;
                    recording.total_monotonic_queueing_time += monotonic_queueing;
                    recording.num_monotonic_served += 1;
                    newest_served = Some(index);
                }
            }
            if !served_and_incomplete.contains(&n64(queue[index].arrival_time)) {
                recording.total_queueing_time += time - queue[index].arrival_time;
                recording.num_served += 1;
                served_and_incomplete.insert(n64(queue[index].arrival_time));
            }
            if recording.num_monotonic_served < recording.num_served {
                dbg!(
                    &queue,
                    &service,
                    num_servers,
                    next_arrival_time,
                    &served_and_incomplete,
                    newest_served,
                    num_arrivals,
                    index,
                    recording.num_served,
                    recording.num_monotonic_served,
                );
            }
            assert!(recording.num_monotonic_served >= recording.num_served);
        }
        let next_event_time = next_arrival_time.min(time + min_service);
        let next_event_duration = next_event_time - time;
        time = next_event_time;

        let mut removal_indexes = vec![];
        for &index in &service {
            let job = &mut queue[index];
            job.remaining_service -= next_event_duration;
        }
        if time < next_arrival_time {
            for &index in &service {
                let job = &mut queue[index];
                if job.remaining_service < EPSILON {
                    removal_indexes.push(index);
                }
            }
            if removal_indexes.len() > 1 {
                removal_indexes.sort();
            }
            for index in removal_indexes.into_iter().rev() {
                let job = queue.remove(index);
                if job.remaining_service < -EPSILON {
                    dbg!(&queue);
                    dbg!(&job);
                    assert!(job.remaining_service >= -EPSILON);
                }
                newest_served = newest_served.expect("Nonempty").checked_sub(1);

                // Stats
                let response_time = time - job.arrival_time;
                recording.total_response_time += response_time;
                recording.num_completed += 1;
                let was_present = served_and_incomplete.remove(&n64(job.arrival_time));
                assert!(was_present);
                num_completed += 1;
            }
        } else {
            next_arrival_time = time + arrival_dist.sample(&mut rng);
            let (n, service) = dist.sample(&mut rng);
            let new_job = Job {
                arrival_time: time,
                remaining_service: service,
                num_servers: n,
            };
            queue.push(new_job);
            num_arrivals += 1;
        }
    }
    if queue.len() > 100 {
        println!("Warning: {} jobs left incomplete. Results not accurate.", queue.len());
    }
    recording.output()
}

fn main() {
    let rho = 0.9;
    //Smaller are faster, more common
    let dist_list = vec![(1, 5.0, 0.9), (2, 2.0, 0.09), (4, 0.5, 0.01)];
    let num_servers = 4;

    let dist = Dist::new(&dist_list);
    println!(
        "E[S] {}, E[service] {}",
        dist.mean_size(),
        dist.mean_service_time()
    );
    let num_jobs = 1_000_000;
    let seed = 0;
    println!(
        "num_jobs {}, num_servers {}, rho {}, E[T_Q^MG1] {}, dist {:?}",
        num_jobs,
        num_servers,
        rho,
        rho * (dist.mean_size_excess()/ num_servers as f64) / (1.0 - rho),
        dist_list
    );
    let policies = vec![Policy::FirstFit, Policy::FCFS, Policy::ServerFilling];
    for policy in &policies {
        let results = simulate(policy, dist.clone(), num_servers, rho, num_jobs, seed);
        println!(
            "{:?},{},{},{}",
            policy,
            results.mean_response_time,
            results.mean_queueing_time,
            results.mean_monotonic_queueing_time
        );
    }
}
