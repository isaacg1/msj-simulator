use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::weighted_alias::WeightedAliasIndex;
use rand_distr::Exp;

use std::collections::{HashMap, HashSet};

use std::f64::INFINITY;
const EPSILON: f64 = 1e-6;

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
    fn max_work(&self, num_servers: u64) -> f64 {
        assert!(num_servers.is_power_of_two());
        self.servers_rates_probs
            .iter()
            .for_each(|(n_i, _, _)| assert!(n_i.is_power_of_two()));
        let max_size = self
            .servers_rates_probs
            .iter()
            .map(|&(n_i, mu_i, _)| n_i as f64 / mu_i)
            .max_by_key(|&f| n64(f))
            .expect("Nonempty");
        let log_num_servers = 63 - num_servers.leading_zeros();
        let filling: f64 = (0..log_num_servers)
            .map(|j| {
                let sub_servers = 1 << j;
                self.servers_rates_probs
                    .iter()
                    .map(|&(n_i, mu_i, _)| {
                        if n_i <= sub_servers {
                            sub_servers as f64 / mu_i
                        } else {
                            0.0
                        }
                    })
                    .max_by_key(|&f| n64(f))
                    .unwrap_or(0.0)
            })
            .sum();
        max_size + filling
    }
}

#[derive(Debug)]
enum Policy {
    // Preemptive
    ServerFilling,
    PreemptiveFirstFit,
    MaxWeight,
    LeastServersFirst,
    MostServersFirst,
    // Nonpreemptive
    FCFS,
    BestFit,
    FirstFit,
    EASYBackfilling,
    ConservativeBackfilling,
    RandomizedTimers,
    StallingMaxWeight(f64),
}

impl Policy {
    fn has_shadow(&self) -> bool {
        match self {
            Policy::ServerFilling | Policy::PreemptiveFirstFit | Policy::FCFS => false,
            Policy::MaxWeight | Policy::LeastServersFirst | Policy::MostServersFirst => true,
            _ => unimplemented!(),
        }
    }
    fn serve(
        &self,
        queue: &Vec<Job>,
        num_servers: u64,
        shadow_indices: &mut Vec<usize>,
    ) -> Vec<usize> {
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
                        dbg!(queue, num_servers);
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
            Policy::PreemptiveFirstFit => {
                let mut servers_occupied = 0;
                let mut service = vec![];
                for (i, job) in queue.iter().enumerate() {
                    if servers_occupied + job.num_servers <= num_servers {
                        servers_occupied += job.num_servers;
                        service.push(i);
                        if servers_occupied == num_servers {
                            break;
                        }
                    }
                }
                service
            }
            Policy::MaxWeight => {
                if queue.is_empty() {
                    vec![]
                } else {
                    let mut counts: HashMap<u64, u64> = HashMap::new();
                    for job in queue {
                        *counts.entry(job.num_servers).or_insert(0) += 1;
                    }
                    let (best_n, count) = counts
                        .iter()
                        .max_by_key(|&(n_i, count)| count * num_servers / n_i)
                        .expect("Nonempty");
                    if best_n * count >= num_servers {
                        (0..queue.len())
                            .filter(|&i| queue[i].num_servers == *best_n)
                            .take((num_servers / best_n) as usize)
                            .collect()
                    } else {
                        shadow_indices.sort_by_key(|&i| {
                            let n_i = queue[i].num_servers;
                            let count_i = counts[&n_i];
                            -((count_i * num_servers / n_i) as i64)
                        });
                        let mut servers_occupied = 0;
                        let mut service = vec![];
                        for &i in shadow_indices.iter() {
                            let job = &queue[i];
                            if servers_occupied + job.num_servers <= num_servers {
                                servers_occupied += job.num_servers;
                                service.push(i);
                                if servers_occupied == num_servers {
                                    break;
                                }
                            }
                        }
                        service
                    }
                }
            }
            Policy::LeastServersFirst => {
                let smallest: Vec<usize> = queue
                    .iter()
                    .enumerate()
                    .filter(|(_, job)| job.num_servers == 1)
                    .take(num_servers as usize)
                    .map(|(i, _)| i)
                    .collect();
                if smallest.len() == num_servers as usize {
                    smallest
                } else {
                    shadow_indices.sort_by_key(|&i| {
                        let n_i = queue[i].num_servers;
                        n_i
                    });
                    let mut servers_occupied = 0;
                    let mut service = vec![];
                    for &i in shadow_indices.iter() {
                        let job = &queue[i];
                        if servers_occupied + job.num_servers <= num_servers {
                            servers_occupied += job.num_servers;
                            service.push(i);
                            if servers_occupied == num_servers {
                                break;
                            }
                        }
                    }
                    service
                }
            }
            Policy::MostServersFirst => {
                let largest: Vec<usize> = queue
                    .iter()
                    .enumerate()
                    .filter(|(_, job)| job.num_servers == num_servers)
                    .take(1)
                    .map(|(i, _)| i)
                    .collect();
                if largest.len() == num_servers as usize {
                    largest
                } else {
                    shadow_indices.sort_by_key(|&i| {
                        let n_i = queue[i].num_servers;
                        -(n_i as i64)
                    });
                    let mut servers_occupied = 0;
                    let mut service = vec![];
                    for &i in shadow_indices.iter() {
                        let job = &queue[i];
                        if servers_occupied + job.num_servers <= num_servers {
                            servers_occupied += job.num_servers;
                            service.push(i);
                            if servers_occupied == num_servers {
                                break;
                            }
                        }
                    }
                    service
                }
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
impl Results {
    fn new_failure() -> Results {
        Results {
            mean_response_time: INFINITY,
            mean_queueing_time: INFINITY,
            mean_monotonic_queueing_time: INFINITY,
        }
    }
}

fn simulate(
    policy: &Policy,
    dist: Dist,
    num_servers: u64,
    rho: f64,
    num_jobs: u64,
    seed: u64,
    kill_above: usize,
) -> Results {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut queue: Vec<Job> = Vec::new();
    let mut shadow_indices: Vec<usize> = Vec::new();
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
    while num_completed < num_jobs || queue.len() > 0 {
        if queue.len() > kill_above {
            return Results::new_failure();
        }
        let service = policy.serve(&queue, num_servers, &mut shadow_indices);
        if debug {
            dbg!(&queue, &service,);
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
                if policy.has_shadow() {
                    shadow_indices = shadow_indices
                        .into_iter()
                        .filter_map(|i| {
                            if i < index {
                                Some(i)
                            } else if i == index {
                                None
                            } else {
                                Some(i - 1)
                            }
                        })
                        .collect();
                }
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
            if num_arrivals < num_jobs {
                let (n, service) = dist.sample(&mut rng);
                let new_job = Job {
                    arrival_time: time,
                    remaining_service: service,
                    num_servers: n,
                };
                queue.push(new_job);
                if policy.has_shadow() {
                    shadow_indices.push(queue.len() - 1);
                }
                num_arrivals += 1;
            }
        }
    }
    if queue.len() > 100 {
        println!(
            "Warning: {} jobs left incomplete. Results not accurate.",
            queue.len()
        );
    }
    recording.output()
}

fn gaps() {
    let rho = 0.9975;
    let num_servers = 8;
    let num_jobs = 40_000_000;
    let seed = 0;
    println!("weight,gap,SF,PFF,MG1,MG1+w_max");
    for log_weight in 1..10 {
        let weight = 1.0 / (1 << log_weight) as f64;
        let dist_list = vec![(1, 1.0, weight), (num_servers, 1e10, 1.0 - weight)];
        let dist = Dist::new(&dist_list);
        let result_PFF = simulate(
            &Policy::PreemptiveFirstFit,
            dist.clone(),
            num_servers,
            rho,
            num_jobs,
            seed,
            50000,
        );
        let result_SF = simulate(
            &Policy::ServerFilling,
            dist.clone(),
            num_servers,
            rho,
            num_jobs,
            seed,
            50000,
        );
        let gap = result_PFF.mean_response_time - result_SF.mean_response_time;
        let mg1 = rho * (dist.mean_size_excess() / num_servers as f64) / (1.0 - rho);
        println!(
            "{},{},{},{},{},{}",
            weight,
            gap,
            result_PFF.mean_response_time,
            result_SF.mean_response_time,
            mg1,
            mg1 + dist.max_work(num_servers),
        );
    }
}

fn many() {
    let rho = 0.99;
    for setting in 0..=5 {
        println!("setting {}", setting);
        let (num_servers, dist_list) = match setting {
            0 => (4, vec![(1, 5.0, 0.9), (2, 2.0, 0.09), (4, 0.5, 0.01)]),
            1 => (
                4,
                vec![
                    (1, 5.0, 1.0 / 3.0),
                    (2, 2.0, 1.0 / 3.0),
                    (4, 0.5, 1.0 / 3.0),
                ],
            ),
            2 => (
                4,
                vec![
                    (1, 1.0, 1.0 / 3.0),
                    (2, 1.0, 1.0 / 3.0),
                    (4, 1.0, 1.0 / 3.0),
                ],
            ),
            3 => (
                4,
                vec![
                    (1, 1.0, 1.0 / 3.0),
                    (2, 2.0, 1.0 / 3.0),
                    (4, 4.0, 1.0 / 3.0),
                ],
            ),
            4 => (
                4,
                vec![
                    (1, 1.0, 4.0 / 7.0),
                    (2, 8.0, 2.0 / 7.0),
                    (4, 64.0, 1.0 / 7.0),
                ],
            ),
            5 => (8, vec![(1, 1.0, 1.0 / 100.0), (8, 1000000.0, 99.0 / 100.0)]),
            _ => unimplemented!(),
        };

        let dist = Dist::new(&dist_list);
        println!(
            "E[s] {}, E[service] {}",
            dist.mean_size(),
            dist.mean_service_time()
        );
        let num_jobs = 10_000_000;
        let seed = 0;
        println!(
            "num_jobs {}, num_servers {}, rho {}, e[t_q^mg1] {}, w_max {}, dist {:?}",
            num_jobs,
            num_servers,
            rho,
            rho * (dist.mean_size_excess() / num_servers as f64) / (1.0 - rho),
            dist.max_work(num_servers),
            dist_list
        );
        let policies = vec![
            Policy::ServerFilling,
            Policy::FCFS,
            Policy::PreemptiveFirstFit,
            Policy::MaxWeight,
            Policy::LeastServersFirst,
            Policy::MostServersFirst,
        ];
        for policy in &policies {
            let results = simulate(policy, dist.clone(), num_servers, rho, num_jobs, seed, 5000);
            println!(
                "{:?},{},{},{}",
                policy,
                results.mean_response_time,
                results.mean_queueing_time,
                results.mean_monotonic_queueing_time
            );
        }
        println!();
    }
}

fn plots() {
    //let num_servers = 4;
    let seed = 0;

    let policies = vec![
        Policy::ServerFilling,
        /*
        Policy::FCFS,
        //Policy::PreemptiveFirstFit,
        Policy::MaxWeight,
        Policy::LeastServersFirst,
        Policy::MostServersFirst,
        */
    ];
    let rhos = vec![
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.72,
        0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94,
        0.96, /*0.903, 0.906, 0.91, 0.913, 0.916, 0.92,
              0.923, 0.926, 0.93, 0.933, 0.936, 0.94, 0.943, 0.946, 0.95, 0.953, 0.956, 0.96, 0.97,
              0.973, 0.976, 0.98, 0.983, 0.986, 0.99, 0.993, 0.996,*/
    ];
    //let rhos = vec![0.86, 0.88, 0.9, 0.92, 0.94, 0.96];
    //let num_jobs = 1e8 as u64;

    //println!("num_servers {}", num_servers,);
    //println!("num_jobs {}", num_jobs);
    println!(
        "let num_jobs = if rho < 0.85 {{
        1e7
    }} else {{
        1e8
    }} as usize;"
    );

    let dist_lists = vec![vec![(1, 1.0 / 2.0, 1.0 / 2.0), (4, 2.0 / 3.0, 1.0 / 2.0)]];
    for dist_list in &dist_lists {
        let dist = Dist::new(dist_list);
        println!(
            "E[s] {}, E[service] {}, dist {:?}",
            dist.mean_size(),
            dist.mean_service_time(),
            dist_list,
        );
    }
    print!("rho;");
    for policy in &policies {
        print!("{:?} s_1 < s_4;", policy);
    }
    for policy in &policies {
        print!("{:?} s_4 < s_1;", policy);
    }
    println!();
    for rho in rhos.clone() {
        print!("{};", rho);
        let num_jobs = if rho < 0.85 { 1e7 } else { 1e8 } as u64;
        for num_servers_exp in vec![1, 2, 3, 4] {
            let num_servers = 1 << num_servers_exp;
            let dist_list = (0..=num_servers_exp)
                .map(|i| (1 << i, (1 << i) as f64/num_servers as f64, 1.0 / (num_servers_exp + 1) as f64))
                .collect();
            let dist = Dist::new(&dist_list);
            for policy in &policies {
                let results =
                    simulate(policy, dist.clone(), num_servers, rho, num_jobs, seed, 5000);
                print!("{};", results.mean_response_time);
            }
        }
        println!();
    }
}
fn main() {
    let kind = 2;
    match kind {
        0 => many(),
        1 => gaps(),
        2 => plots(),
        _ => unimplemented!(),
    }
}
