const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const { GoalNear, GoalBlock } = goals; // GoalBlock può essere utile
const { Vec3 } = require('vec3'); // Utile per manipolare le posizioni
const viewer = require('prismarine-viewer').mineflayer;

const BOT_USERNAME = 'TreeTerminator';
const TARGET_X = 52; // Sostituisci
const TARGET_Y = 86; // Sostituisci
const TARGET_Z = 85; // Sostituisci
const MAX_TOTAL_BLOCKS_TO_CHOP = 500; // Limite totale di blocchi prima di fermarsi
const SEARCH_RADIUS = 64; // Raggio per trovare il prossimo albero

const bot = mineflayer.createBot({
  host: 'localhost',
  port: 49289,
  username: BOT_USERNAME,
});

bot.loadPlugin(pathfinder);

let mcData;
let totalBlocksChopped = 0;
let isCurrentlyChoppingTree = false; // Flag per indicare se si sta lavorando su un albero

// Tipi di tronchi
const LOG_TYPES = [
  'oak_log', 'birch_log', 'spruce_log', 'jungle_log',
  'acacia_log', 'dark_oak_log', 'mangrove_log', 'cherry_log',
  // 'crimson_stem', 'warped_stem' // Richiedono attrezzi specifici
];

bot.once('spawn', () => {
  mcData = require('minecraft-data')(bot.version);
  const movements = new Movements(bot, mcData);
  movements.canDig = true; // Assicurati che il pathfinder sappia che può scavare
  movements.allow1by1towers = true; // Permetti di fare torri 1x1 se necessario (con cautela)
  bot.pathfinder.setMovements(movements);

  console.log(`[${BOT_USERNAME}] Spawned. Teleporting...`);
  // bot.chat(`/tp ${BOT_USERNAME} ${TARGET_X} ${TARGET_Y} ${TARGET_Z}`);

  setTimeout(() => {
    console.log(`[${BOT_USERNAME}] Teleported. Current pos: ${bot.entity.position}`);
    bot.chat("Pronto a tagliare alberi interi!");
    try {
      viewer(bot, { port: 3007, firstPerson: true });
      console.log(`[${BOT_USERNAME}] Viewer started on http://localhost:3007`);
    } catch (e) { console.warn(`[${BOT_USERNAME}] Viewer failed: ${e.message}`); }

    findAndChopNextTree();
  }, 2500);
});

async function findAndChopNextTree() {
  if (isCurrentlyChoppingTree) {
    console.log(`[${BOT_USERNAME}] findAndChopNextTree: Already chopping a tree. Waiting.`);
    return;
  }
  if (totalBlocksChopped >= MAX_TOTAL_BLOCKS_TO_CHOP) {
    bot.chat(`Ho tagliato ${totalBlocksChopped} blocchi. Missione compiuta!`);
    console.log(`[${BOT_USERNAME}] Max blocks chopped. Stopping.`);
    return;
  }

  console.log(`[${BOT_USERNAME}] Cerco il prossimo albero...`);
  const treeBase = findLowestLogBlockOfTree();

  if (!treeBase) {
    bot.chat("🔍 Nessun albero trovato nelle vicinanze. Aspetto e riprovo.");
    console.log(`[${BOT_USERNAME}] Nessun albero trovato. Riprovo tra 15s.`);
    setTimeout(findAndChopNextTree, 15000);
    return;
  }

  bot.chat(`🪓 Trovato albero a ${treeBase.position}. Mi avvicino...`);
  console.log(`[${BOT_USERNAME}] Trovato albero con base a ${treeBase.position}.`);

  try {
    await bot.pathfinder.goto(new GoalNear(treeBase.position.x, treeBase.position.y, treeBase.position.z, 1));
    console.log(`[${BOT_USERNAME}] Raggiunta la base dell'albero.`);
    isCurrentlyChoppingTree = true;
    await excavateTreeRecursive(treeBase); // Inizia a scavare l'albero
    isCurrentlyChoppingTree = false;
    console.log(`[${BOT_USERNAME}] Finito di tagliare l'albero corrente.`);
    bot.chat("✅ Albero tagliato completamente!");
  } catch (err) {
    console.error(`[${BOT_USERNAME}] Errore durante il raggiungimento o il taglio dell'albero: ${err.message}`);
    bot.chat("⚠️ Problema con l'albero, ne cerco un altro.");
    isCurrentlyChoppingTree = false;
  }

  // Cerca il prossimo albero dopo un breve ritardo
  setTimeout(findAndChopNextTree, 1000);
}

function findLowestLogBlockOfTree() {
  // Trova tutti i blocchi di log nel raggio
  const allLogs = bot.findBlocks({
    matching: block => LOG_TYPES.includes(block.name),
    maxDistance: SEARCH_RADIUS,
    count: 200 // Cerca un buon numero di blocchi per trovare una base
  });

  if (allLogs.length === 0) return null;

  // Filtra per trovare solo i blocchi di log che hanno aria o foglie sotto (probabili basi di alberi)
  // o che sono i più bassi di un gruppo.
  let lowestLog = null;
  for (const logPos of allLogs) {
    const block = bot.blockAt(logPos);
    if (!block) continue;

    const blockBelow = bot.blockAt(logPos.offset(0, -1, 0));
    // Un blocco di base di un albero di solito ha terra/erba sotto, o è il più basso di un gruppo
    if (blockBelow && (blockBelow.name === 'air' || LOG_TYPES.includes(blockBelow.name) || blockBelow.name.includes('leaves'))) {
        // Se sotto c'è aria o un altro log, potrebbe non essere la base effettiva, ma parte di un albero caduto o strano.
        // Per ora, lo consideriamo valido se è il più basso trovato finora.
    }

    if (!lowestLog || block.position.y < lowestLog.position.y) {
        // Verifica anche che non ci sia un altro log direttamente sopra,
        // per evitare di iniziare da un ramo laterale basso come se fosse la base.
        // Questa logica può essere migliorata per alberi complessi.
        const blockAbove = bot.blockAt(logPos.offset(0, 1, 0));
        if (blockAbove && LOG_TYPES.includes(blockAbove.name)) {
             lowestLog = block;
        } else if (!blockAbove || !LOG_TYPES.includes(blockAbove.name)) {
            // Se non c'è un log sopra, potrebbe essere la cima di un piccolo albero
            // o un albero con un solo blocco. In quel caso va bene.
            lowestLog = block;
        }
    }
  }
  // Preferisci un blocco di log che ha un blocco non-log sotto (es. dirt, grass)
  const potentialBases = allLogs.map(pos => bot.blockAt(pos)).filter(block => {
      if (!block) return false;
      const blockBelow = bot.blockAt(block.position.offset(0, -1, 0));
      return blockBelow && !LOG_TYPES.includes(blockBelow.name) && blockBelow.name !== 'air';
  });

  if (potentialBases.length > 0) {
      potentialBases.sort((a,b) => bot.entity.position.distanceTo(a.position) - bot.entity.position.distanceTo(b.position));
      console.log(`[DEBUG] Trovate ${potentialBases.length} basi potenziali. Scelta la più vicina: ${potentialBases[0].position}`);
      return potentialBases[0];
  } else if (lowestLog) {
      console.log(`[DEBUG] Nessuna base ideale trovata. Scelgo il log più basso: ${lowestLog.position}`);
      return lowestLog; // Fallback al log più basso trovato se nessuna base "ideale"
  }

  return null;
}


async function excavateTreeRecursive(blockToChop, visited = new Set()) {
  if (!blockToChop || !LOG_TYPES.includes(blockToChop.name)) {
    return;
  }
  if (totalBlocksChopped >= MAX_TOTAL_BLOCKS_TO_CHOP) {
    return; // Limite raggiunto
  }

  const blockPosStr = blockToChop.position.toString();
  if (visited.has(blockPosStr)) {
    return; // Già visitato o in corso di visita
  }
  visited.add(blockPosStr);

  // Assicurati che il bot sia abbastanza vicino per rompere il blocco
  // Pathfinder potrebbe non portare il bot esattamente al blocco se è in aria
  const distance = bot.entity.position.distanceTo(blockToChop.position.offset(0.5, 0.5, 0.5));
  if (distance > 4.5) { // 4.5 è la portata massima di scavo circa
    console.log(`[${BOT_USERNAME}] excavate: Blocco ${blockToChop.name} a ${blockToChop.position} troppo lontano (${distance.toFixed(1)}m). Mi avvicino...`);
    try {
      // Usare GoalBlock per raggiungere specificamente il blocco o la sua adiacenza
      // A volte GoalNear è meglio se il blocco è difficile da pathare direttamente (es. in alto)
      await bot.pathfinder.goto(new GoalNear(blockToChop.position.x, blockToChop.position.y, blockToChop.position.z, 1.5));
    } catch (err) {
      console.error(`[${BOT_USERNAME}] excavate: Impossibile avvicinarsi a ${blockToChop.position}: ${err.message}`);
      visited.delete(blockPosStr); // Rimuovi dalla visita così può essere ritentato
      return; // Non posso raggiungere questo blocco, passo oltre (o l'albero)
    }
  }

  // Controlla se il blocco esiste ancora (potrebbe essere stato rotto da qualcos'altro o da un ciclo precedente)
  const currentBlock = bot.blockAt(blockToChop.position);
  if (!currentBlock || !LOG_TYPES.includes(currentBlock.name)) {
    console.log(`[${BOT_USERNAME}] excavate: Blocco ${blockToChop.name} a ${blockToChop.position} non più presente o tipo errato.`);
    return;
  }


  console.log(`[${BOT_USERNAME}] excavate: Tento di scavare ${currentBlock.name} a ${currentBlock.position}`);
  try {
    // Equipaggia un'ascia se disponibile (aggiunta opzionale, ma altamente raccomandata)
    await equipBestAxe();

    await bot.dig(currentBlock, true); // true per forceLook
    totalBlocksChopped++;
    console.log(`[${BOT_USERNAME}] excavate: Blocco ${currentBlock.name} scavato! Totale: ${totalBlocksChopped}`);
    bot.chat(`Blocco #${totalBlocksChopped} tagliato.`);

    if (totalBlocksChopped >= MAX_TOTAL_BLOCKS_TO_CHOP) {
      console.log(`[${BOT_USERNAME}] Limite di ${MAX_TOTAL_BLOCKS_TO_CHOP} blocchi raggiunto durante excavate.`);
      bot.chat("Limite blocchi raggiunto!");
      return;
    }

    // Breve pausa per permettere al server di aggiornare e ai blocchi di cadere
    await bot.waitForTicks(5);

  } catch (err) {
    console.error(`[${BOT_USERNAME}] excavate: Errore scavando ${currentBlock.name} a ${currentBlock.position}: ${err.message}`);
    // Potrebbe essere che il blocco è sparito o il bot si è mosso. Continua con gli altri.
    visited.delete(blockPosStr);
    return; // Non continuare a cercare adiacenti se questo fallisce
  }

  // Cerca blocchi di tronco adiacenti (su, giù, e lati)
  // Per gli alberi normali, principalmente sopra. Per alberi grandi (dark oak, jungle), anche ai lati.
  const offsets = [
    new Vec3(0, 1, 0),  // Sopra
    // new Vec3(0, -1, 0), // Sotto (generalmente non necessario se si parte dal basso)
    // Per alberi più grandi, considera anche i lati:
    new Vec3(1, 0, 0), new Vec3(-1, 0, 0),
    new Vec3(0, 0, 1), new Vec3(0, 0, -1),
    // E anche diagonali per alberi molto spessi (es. dark oak 2x2)
    new Vec3(1, 0, 1), new Vec3(1, 0, -1),
    new Vec3(-1, 0, 1), new Vec3(-1, 0, -1),
    // E anche blocchi sopra i laterali (per rami)
    new Vec3(1, 1, 0), new Vec3(-1, 1, 0),
    new Vec3(0, 1, 1), new Vec3(0, 1, -1),
  ];

  for (const offset of offsets) {
    const nextPos = currentBlock.position.plus(offset);
    const nextBlock = bot.blockAt(nextPos);
    if (nextBlock && LOG_TYPES.includes(nextBlock.name) && !visited.has(nextPos.toString())) {
      // Ricorsione: aspetta che la chiamata ricorsiva finisca prima di passare al prossimo adiacente
      await excavateTreeRecursive(nextBlock, visited);
      if (totalBlocksChopped >= MAX_TOTAL_BLOCKS_TO_CHOP) break; // Controlla di nuovo il limite
    }
  }
}

async function equipBestAxe() {
    const axes = bot.inventory.items().filter(item => item.name.includes('_axe'));
    if (axes.length === 0) {
        // console.log("[DEBUG] No axe in inventory.");
        return; // Nessuna ascia
    }
    // Semplice: equipaggia la prima ascia trovata. Per "migliore" servirebbe logica sui materiali.
    try {
        const bestAxe = axes[0]; // Semplificazione
        // Controlla se è già equipaggiata per evitare spam di equip
        if (!bot.heldItem || bot.heldItem.type !== bestAxe.type) {
            await bot.equip(bestAxe, 'hand');
            // console.log(`[DEBUG] Equipped ${bestAxe.name}.`);
        }
    } catch (err) {
        console.error(`[${BOT_USERNAME}] Errore equipaggiando l'ascia: ${err.message}`);
    }
}

// Gestione errori e disconnessione
bot.on('kicked', (reason) => {
  console.error(`[${BOT_USERNAME}] Kicked: ${reason}`);
  isCurrentlyChoppingTree = false;
});
bot.on('error', (err) => {
  console.error(`[${BOT_USERNAME}] Error: `, err);
  isCurrentlyChoppingTree = false;
});
bot.on('end', (reason) => {
  console.log(`[${BOT_USERNAME}] Disconnected: ${reason}`);
  isCurrentlyChoppingTree = false;
});
process.on('SIGINT', () => {
  console.log(`[${BOT_USERNAME}] SIGINT received, quitting...`);
  bot.quit();
  process.exit(0);
});