use std::ffi::CStr;

use cudarc::driver::{result as cuda_driver_result, sys as cuda_driver_sys};

pub(super) fn pin_rank_worker_thread(rank: usize, device_ordinal: usize) {
    let label = format!("DeepSeek rank worker {rank}");
    pin_rank_thread(&label, rank, device_ordinal);
}

fn pin_rank_thread(label: &str, rank: usize, device_ordinal: usize) {
    let allowed_cpus = current_allowed_cpus()
        .unwrap_or_else(|| panic!("failed to read CPU affinity mask for {label}"));
    assert!(
        !allowed_cpus.is_empty(),
        "empty CPU affinity mask for {label}"
    );

    let target = numa_target_for_device(device_ordinal).unwrap_or_else(|err| panic!("{err}"));
    let target_cpus = target
        .cpus
        .iter()
        .copied()
        .filter(|cpu| allowed_cpus.contains(cpu))
        .collect::<Vec<_>>();
    assert!(
        !target_cpus.is_empty(),
        "NUMA node {} CPU list for CUDA device {} ({}) has no overlap with allowed CPU mask",
        target.numa_node,
        device_ordinal,
        target.pci_bus_id
    );
    let cpu = target_cpus[rank % target_cpus.len()];

    assert!(
        pin_current_thread_to_cpu(cpu),
        "failed to pin {label} to CPU {cpu}"
    );
    log::info!(
        "pinned {label} for CUDA device {device_ordinal} ({}) NUMA{} to CPU {cpu}",
        target.pci_bus_id,
        target.numa_node
    );
}

fn current_allowed_cpus() -> Option<Vec<usize>> {
    unsafe {
        let mut allowed: libc::cpu_set_t = std::mem::zeroed();
        let set_size = std::mem::size_of::<libc::cpu_set_t>();
        if libc::sched_getaffinity(0, set_size, &mut allowed) != 0 {
            return None;
        }

        let mut cpus = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &allowed) {
                cpus.push(cpu);
            }
        }
        Some(cpus)
    }
}

struct NumaTarget {
    pci_bus_id: String,
    numa_node: usize,
    cpus: Vec<usize>,
}

fn numa_target_for_device(device_ordinal: usize) -> Result<NumaTarget, String> {
    let pci_bus_id = cuda_pci_bus_id(device_ordinal).map_err(|err| {
        format!("failed to read PCI bus id for CUDA device {device_ordinal}: {err}")
    })?;
    let numa_node = pci_numa_node(&pci_bus_id).ok_or_else(|| {
        format!("failed to read NUMA node for CUDA device {device_ordinal} ({pci_bus_id})")
    })?;
    let cpulist =
        std::fs::read_to_string(format!("/sys/devices/system/node/node{numa_node}/cpulist"))
            .map_err(|err| {
                format!(
                    "failed to read CPU list for CUDA device {device_ordinal} ({pci_bus_id}) NUMA{numa_node}: {err}"
                )
            })?;
    let cpus = parse_cpu_list(cpulist.trim()).ok_or_else(|| {
        format!(
            "failed to parse CPU list for CUDA device {device_ordinal} ({pci_bus_id}) NUMA{numa_node}: {cpulist:?}"
        )
    })?;
    if cpus.is_empty() {
        return Err(format!(
            "empty CPU list for CUDA device {device_ordinal} ({pci_bus_id}) NUMA{numa_node}"
        ));
    }
    Ok(NumaTarget {
        pci_bus_id,
        numa_node,
        cpus,
    })
}

fn cuda_pci_bus_id(device_ordinal: usize) -> Result<String, String> {
    let mut buf = [0i8; 32];
    cuda_driver_result::init().map_err(|err| format!("{err:?}"))?;
    let device =
        cuda_driver_result::device::get(device_ordinal as i32).map_err(|err| format!("{err:?}"))?;
    unsafe {
        cuda_driver_sys::cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as i32, device)
            .result()
            .map_err(|err| format!("{err:?}"))?;
        CStr::from_ptr(buf.as_ptr())
            .to_str()
            .map(|bus_id| bus_id.to_ascii_lowercase())
            .map_err(|err| err.to_string())
    }
}

fn pci_numa_node(pci_bus_id: &str) -> Option<usize> {
    let raw =
        std::fs::read_to_string(format!("/sys/bus/pci/devices/{pci_bus_id}/numa_node")).ok()?;
    let node = raw.trim().parse::<isize>().ok()?;
    usize::try_from(node).ok()
}

fn parse_cpu_list(cpulist: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in cpulist.split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            let start = start.parse::<usize>().ok()?;
            let end = end.parse::<usize>().ok()?;
            if start > end {
                return None;
            }
            cpus.extend(start..=end);
        } else {
            cpus.push(part.parse::<usize>().ok()?);
        }
    }
    Some(cpus)
}

fn pin_current_thread_to_cpu(cpu: usize) -> bool {
    unsafe {
        let mut target: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut target);
        libc::CPU_SET(cpu, &mut target);
        let set_size = std::mem::size_of::<libc::cpu_set_t>();
        libc::pthread_setaffinity_np(libc::pthread_self(), set_size, &target) == 0
    }
}
