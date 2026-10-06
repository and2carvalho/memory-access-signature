> **Nota editorial (adicionada em 2026-10-06, na preparação deste repositório público).**
> O texto do pré-registro abaixo é reproduzido sem alterações. Duas observações:
>
> 1. **Precedência temporal.** A frase "este arquivo precede, no tempo e no histórico do
>    repositório, o artefato `output/crossdomain.json`" não é verificável pelo histórico
>    de versões: o pré-registro, o driver e os resultados foram versionados no mesmo commit
>    do repositório de pesquisa original (privado), em 2026-08-03. A anterioridade das
>    hipóteses é, portanto, uma declaração do autor, e não uma evidência independente
>    (ver a seção *Threats to validity* em `docs/cross-domain-results.md`).
> 2. **Caminhos.** Este documento ficava em `analysis/PREREGISTRATION.md`; a tradução para
>    o inglês está em `docs/preregistration.md`. O manuscrito citado ao final
>    (`paper/paper.md`) não é distribuído neste repositório.

---

# Pré-registração — Validação cross-domínio da assinatura (D, δ)

**Escrito ANTES de executar `src/build_crossdomain.ts`.** Este documento fixa as
hipóteses e as previsões numéricas *antes* de qualquer medição das cargas não-sort, para
eliminar a crítica de interpretação pós-hoc. A ordem importa: este arquivo precede, no
tempo e no histórico do repositório, o artefato `output/crossdomain.json`.

## Tese sob teste

A assinatura `(D, δ)` não reconhece *algoritmos*; ela identifica **famílias fundamentais
de padrão de acesso à memória**, das quais um algoritmo de ordenação é apenas uma
instância. Se isso for verdade, cargas **fora de ordenação** que instanciam a mesma
família devem cair na **mesma região** do plano `(log D, δ)` — usando os **mesmos
limiares** calibrados no corpus de sorts (fronteira de carga inflada D≈5; fronteira de
sinal δ=0) e a **mesma referência** de densidade `E(ref)` = Merge/aleatório no mesmo n.

Nenhum limiar é reajustado para o novo domínio. Este é um teste de **transferência**.

## As quatro famílias e suas instâncias não-sort

| Família | Instância sort (corpus atual) | Instância não-sort (este teste) |
|---|---|---|
| Forward scan (δ>0) | Merge, Radix | `linear_reduce` (varredura/redução) |
| Root return (δ<0) | Heapsort | `tree_reduce` (fold bottom-up árvore→raiz) |
| Random access (δ≈0) | *nenhuma* | `hash_probe` (probes aleatórios) |
| Quadratic overload (D≫5) | Insertion, QuickSort naive | `self_join` (nested-loop O(n²)) |

## Hipóteses pré-registradas

**H1 — Forward scan ⟹ δ>0.**
Varreduras progressivas produzem deriva positiva.
*Previsão:* `linear_reduce` tem **δ > +40‰** e **D ≪ 1** (carga O(n), leve).

**H2 — Root-return ⟹ δ<0.**
Travessias que retornam à origem/raiz produzem deriva negativa.
*Previsão:* `tree_reduce` tem **δ < 0** (sinal negativo é o discriminador, não a magnitude).

**H3 — Cargas quadráticas ⟹ D>5, independentemente da família algorítmica.**
A fronteira de carga inflada detecta degeneração/explosão quadrática mesmo *fora* de
ordenação e mesmo quando o O(n²) é **estrutural** (não disparado por dado adversário).
*Previsão:* `self_join` tem **D > 5** (esperado D ~ 10²–10³ em n=10⁴). Nenhuma previsão
de δ é registrada para H3: a família "quadratic overload" é definida por D, não por δ.

**H4 — Random probing ⟹ δ≈0.**
Acesso aleatório sem tendência temporal produz deriva próxima de zero.
*Previsão:* `hash_probe` tem **|δ| < 10‰** e **D ≪ 1**.

## Critério de decisão (fixado a priori)

- H1 confirmada se δ(linear_reduce) > +40 **e** D < 1.
- H2 confirmada se δ(tree_reduce) < 0.
- H3 confirmada se D(self_join) > 5.
- H4 confirmada se |δ(hash_probe)| < 10.

Uma hipótese **falsificada** é um resultado científico, não uma falha: dispara a pergunta
"por quê?" e a busca pela estrutura de acesso que explica o desvio. O placar (confirmadas
× falsificadas) será reportado tal como medido, sem reajuste retroativo das previsões.

## Protocolo de medição

- Sizes n ∈ {10³, 2·10³, 5·10³, 10⁴, 2·10⁴}; `self_join` (O(n²)) limitado por `--quad-cap`.
- Conteúdo do vetor: padrão `aleatorio`, seed base `20260731` (mesma família de seeds do
  atlas de sorts). Cargas não-sort são, em geral, **independentes do conteúdo** (o padrão
  de acesso vem do algoritmo, não do dado) — anotado como distinção frente aos sorts.
- `E(ref)` = eventos de Merge/aleatório no mesmo n (recomputado com a mesma seed → idêntico
  ao atlas). `D = E(carga)/E(ref)`; `δ = indexTrajectoryDrift(...)` de `src/signature.ts`.
- Saída consolidada em `output/crossdomain.json`; nenhum traço bruto versionado.

_Autor: André Cunha Antero de Carvalho (PPGCO/UFU). Assinatura (D, δ) — paper `paper/paper.md`._
